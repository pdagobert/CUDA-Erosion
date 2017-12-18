#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>

#include <glm/glm.hpp>
#include <Simplex.h>

std::vector< float > generateNoiseTexture( int size )
{
    std::vector< float > heightmap;
    heightmap.reserve( size * size );

    for( int y = 0; y < size; y++ )
    {
        for( int x = 0; x < size; x++ )
        {
            glm::vec2 position( x, y );
            position /= size;

            float height = Simplex::fBm( position, 8, 2.0f, 0.5f );
            heightmap.push_back( height );
        }
    }

    return heightmap;
}

// https://github.com/RolandR/glterrain/blob/master/js/terrain.js
void erode( std::vector< float >& heightmap, int size, int iterations )
{
    const float Scale = 1.0f;
    const float Erosion = 0.0005 * Scale;
    const float Deposition = 0.0000002 * Scale;
    const float Evaporation = 0.9f;

    std::vector< float > water( heightmap.size(), 1.0f );
    std::vector< float > dWater( heightmap.size(), 0 );
    std::vector< float > dheightV( heightmap.size(), 0 );

    for( int i = 0; i < iterations; i++ )
    {
        // ignore bounds for now
        for( int y = 1; y < size - 2; y++ )
        {
            for( int x = 1; x < size - 2; x++ )
            {
                int index = x + y * size;

                float m = heightmap[ index ];
                float dtl = heightmap[ x - 1 + ( y + 1 ) * size ];
                dtl = glm::max( m - dtl, 0.0f );
                float dt = heightmap[ x + ( y + 1 ) * size ];
                dt = glm::max( m - dt, 0.0f );
                float dtr = heightmap[ x + 1 + ( y + 1 ) * size ];
                dtr = glm::max( m - dtr, 0.0f );
                float dml = heightmap[ x - 1 + y * size ];
                dml = glm::max( m - dml, 0.0f );
                float dmr = heightmap[ x + 1 + y * size ];
                dmr = glm::max( m - dmr, 0.0f );
                float dbl = heightmap[ x - 1 + ( y - 1 ) * size ];
                dbl = glm::max( m - dbl, 0.0f );
                float db = heightmap[ x + ( y - 1 ) * size ];
                db = glm::max( m - db, 0.0f );
                float dbr = heightmap[ x + 1 + ( y - 1 ) * size ];
                dbr = glm::max( m - dbr, 0.0f );

                float dheight = dtl + dt + dtr + dml + dmr + dbl + db + dbr;

                if( dheight != 0.0f )
                {
                    float w = water[ index ] * Evaporation;
                    float remainingWater = w * 0.0002 / ( dheight * Scale + 1.0f );
                    w -= remainingWater;

                    dWater[ x - 1 + ( y + 1 ) * size ] += dtl / dheight * w;
                    dWater[ x + ( y + 1 ) * size ] += dt / dheight * w;
                    dWater[ x + 1 + ( y + 1 ) * size ] += dtr / dheight * w;
                    dWater[ x - 1 + y * size ] += dml / dheight * w;
                    dWater[ x + 1 + y * size ] += dmr / dheight * w;
                    dWater[ x - 1 + ( y - 1 ) * size ] += dbl / dheight * w;
                    dWater[ x + ( y - 1 ) * size ] += db / dheight * w;
                    dWater[ x + 1 + ( y - 1 ) * size ] += dbr / dheight * w;

                    water[ index ] = 1.0f + remainingWater;
                }

                dheightV[ index ] = dheight;
            }
        }

        for( int y = 1; y < size - 2; y++ )
        {
            for( int x = 1; x < size - 2; x++ )
            {
                int index = x + y * size;

                water[ index ] += dWater[ index ];
                dWater[ index ] = 0.0f;

                float oldHeight = heightmap[ index ];
                heightmap[ index ] += ( -( dheightV[ index ] - 0.005f / Scale ) * water[ index ] ) * Erosion + water[ index ] * Deposition;

                if( oldHeight < heightmap[ index ] )
                    water[ index ] = glm::max( water[index ] - ( heightmap[ index ] - oldHeight ) * 1000.0f, 0.0f );
            }
        }
    }
}

void save( const std::string& fileName, int width, int height, const std::vector< float >& heightmap )
{
    std::ofstream file( fileName, std::ios::binary );
    if( file.fail() )
    {
        std::cout << "cannot save file " << fileName << std::endl;
        return;
    }

    file.write( reinterpret_cast< char* >( &width ), sizeof( int ) );
    file.write( reinterpret_cast< char* >( &height ), sizeof( int ) );
    file.write( reinterpret_cast< const char* >( &heightmap[ 0 ] ), heightmap.size() * sizeof( float ) );
}

void cpuErosion( int size, int iterations )
{
    auto heightmap = generateNoiseTexture( size );

    auto start = std::chrono::high_resolution_clock::now();
    erode( heightmap, size, iterations );

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast< std::chrono::milliseconds >( end - start ).count();

    std::cout << "cpu erosion took " << duration << "ms" << std::endl;

    save( "noiseCPU.raw", size, size, heightmap );
}

__global__
void erodeKernel( float* heightmap, int size, int iterations, float* water, float* tmpWater, float* dheightV )
{
    const float Scale = 1.0f;
    const float Erosion = 0.0005 * Scale;
    const float Deposition = 0.0000002 * Scale;
    const float Evaporation = 0.9f;

    int startx = threadIdx.x + blockIdx.x * blockDim.x;
    int starty = threadIdx.y + blockIdx.y * blockDim.y;
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;

    for( int i = 0; i < iterations; i++ )
    {
        for( int y = 1 + starty; y < size - 1; y += stridey )
        {
            for( int x = 1 + startx; x < size - 1; x += stridex )
            {
                int index = x + y * size;

                float m = heightmap[ index ];

                float dtl = heightmap[ x - 1 + ( y + 1 ) * size ];
                dtl = glm::max( m - dtl, 0.0f );
                float dt = heightmap[ x + ( y + 1 ) * size ];
                dt = glm::max( m - dt, 0.0f );
                float dtr = heightmap[ x + 1 + ( y + 1 ) * size ];
                dtr = glm::max( m - dtr, 0.0f );
                float dml = heightmap[ x - 1 + y * size ];
                dml = glm::max( m - dml, 0.0f );
                float dmr = heightmap[ x + 1 + y * size ];
                dmr = glm::max( m - dmr, 0.0f );
                float dbl = heightmap[ x - 1 + ( y - 1 ) * size ];
                dbl = glm::max( m - dbl, 0.0f );
                float db = heightmap[ x + ( y - 1 ) * size ];
                db = glm::max( m - db, 0.0f );
                float dbr = heightmap[ x + 1 + ( y - 1 ) * size ];
                dbr = glm::max( m - dbr, 0.0f );

                float dheight = dtl + dt + dtr + dml + dmr + dbl + db + dbr;

                if( dheight != 0.0f )
                {
                    float w = water[ index ] * Evaporation;
                    float remainingWater = w * 0.0002 / ( dheight * Scale + 1.0f );
                    w -= remainingWater;

                    // the only place where race condition can occur

                    atomicAdd( &tmpWater[ x - 1 + ( y + 1 ) * size ], dtl / dheight * w );
                    atomicAdd( &tmpWater[ x + ( y + 1 ) * size ], dt / dheight * w );
                    atomicAdd( &tmpWater[ x + 1 + ( y + 1 ) * size ], dtr / dheight * w );
                    atomicAdd( &tmpWater[ x - 1 + y * size ], dml / dheight * w );
                    atomicAdd( &tmpWater[ x + 1 + y * size ], dmr / dheight * w );
                    atomicAdd( &tmpWater[ x - 1 + ( y - 1 ) * size ], dbl / dheight * w );
                    atomicAdd( &tmpWater[ x + ( y - 1 ) * size ], db / dheight * w );
                    atomicAdd( &tmpWater[ x + 1 + ( y - 1 ) * size ], dbr / dheight * w );

                    water[ index ] = 1.0f + remainingWater;
                }

                dheightV[ index ] = dheight;
            }
        }

        __syncthreads();

        for( int y = 1 + starty; y < size - 1; y += stridey )
        {
            for( int x = 1 + startx; x < size - 1; x += stridex )
            {
                int index = x + y * size;

                water[ index ] += tmpWater[ index ];
                tmpWater[ index ] = 0;

                float oldHeight = heightmap[ index ];
                heightmap[ index ] += ( -( dheightV[ index ] - 0.005f / Scale ) * water[ index ] ) * Erosion + water[ index ] * Deposition;

                if( oldHeight < heightmap[ index ] )
                    water[ index ] = glm::max( water[index ] - ( heightmap[ index ] - oldHeight ) * 1000.0f, 0.0f );
            }
        }

        __syncthreads();
    }
}

__global__
void erodeFirstPassKernel( float* __restrict__ heightmap, int size, float* __restrict__ water, float* __restrict__ tmpWater, float* __restrict__ dheightV )
{
    __shared__ float heightmapCache[ 34 * 34 ];

    const float Scale = 1.0f;
    const float Evaporation = 0.9f;

    /*int startx = threadIdx.x + blockIdx.x * blockDim.x;
    int starty = threadIdx.y + blockIdx.y * blockDim.y;
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;

    for( int y = 1 + starty; y < size - 2; y += stridey )
    {
        for( int x = 1 + startx; x < size - 2; x += stridex )
        {*/
            int x2 = threadIdx.x + blockIdx.x * blockDim.x;
            int y2 = threadIdx.y + blockIdx.y * blockDim.y;
            int indexGlobal = x2 + y2 * size;

            int tileSize = 34;

            int x = threadIdx.x + 1;
            int y = threadIdx.y + 1;
            int indexLocal = x + y * tileSize;

            heightmapCache[ indexLocal ] = heightmap[ indexGlobal ];

            if( x2 == 0 || y2 == 0 || x2 == size - 1 || y2 == size - 1 )
                return;

            int offset = 0;

            if( threadIdx.x == 0 )
                offset = -1;

            if( threadIdx.x == 31 )
                offset = 1;

            if( offset != 0 )
            {
                int borderGlobal = x2 + offset + y2 * size;
                int borderLocal = x + offset + y * tileSize;
                heightmapCache[ borderLocal ] = heightmap[ borderGlobal ];
            }

            offset = 0;

            if( threadIdx.y == 0 )
                offset = -1;

            if( threadIdx.y == 31 )
                offset = 1;

            if( offset != 0 )
            {
                int borderGlobal = x2 + ( y2 + offset ) * size;
                int borderLocal = x + ( y + offset ) * tileSize;
                heightmapCache[ borderLocal ] = heightmap[ borderGlobal ];
            }

            int xoffset = 0;
            int yoffset = 0;

            if( threadIdx.x == 0 && threadIdx.y == 0 )
            {
                xoffset = -1;
                yoffset = -1;
            }

            if( threadIdx.x == 31 && threadIdx.y == 0 )
            {
                xoffset = 1;
                yoffset = -1;
            }

            if( threadIdx.x == 0 && threadIdx.y == 31 )
            {
                xoffset = -1;
                yoffset = 1;
            }

            if( threadIdx.x == 31 && threadIdx.y == 31 )
            {
                xoffset = 1;
                yoffset = 1;
            }

            if( xoffset != 0 )
            {
                int borderGlobal = x2 + xoffset + ( y2 + yoffset ) * size;
                int borderLocal = x + xoffset + ( y + yoffset ) * tileSize;
                heightmapCache[ borderLocal ] = heightmap[ borderGlobal ];
            }

            __syncthreads();

            float m = heightmapCache[ x + y * tileSize ];
            float dtl = heightmapCache[ x - 1 + ( y + 1 ) * tileSize ];
            dtl = max( m - dtl, 0.0f );
            float dt = heightmapCache[ x + ( y + 1 ) * tileSize ];
            dt = max( m - dt, 0.0f );
            float dtr = heightmapCache[ x + 1 + ( y + 1 ) * tileSize ];
            dtr = max( m - dtr, 0.0f );
            float dml = heightmapCache[ x - 1 + y * tileSize ];
            dml = max( m - dml, 0.0f );
            float dmr = heightmapCache[ x + 1 + y * tileSize ];
            dmr = max( m - dmr, 0.0f );
            float dbl = heightmapCache[ x - 1 + ( y - 1 ) * tileSize ];
            dbl = max( m - dbl, 0.0f );
            float db = heightmapCache[ x + ( y - 1 ) * tileSize ];
            db = max( m - db, 0.0f );
            float dbr = heightmapCache[ x + 1 + ( y - 1 ) * tileSize ];
            dbr = max( m - dbr, 0.0f );

            float dheight = dtl + dt + dtr + dml + dmr + dbl + db + dbr;

            if( dheight != 0.0f )
            {
                float w = water[ indexGlobal ] * Evaporation;
                float remainingWater = w * 0.0002 / ( dheight * Scale + 1.0f );
                w -= remainingWater;

                // the only place where race condition can occur

                atomicAdd( &tmpWater[ x2 - 1 + ( y2 + 1 ) * size ], dtl / dheight * w );
                atomicAdd( &tmpWater[ x2 + ( y2 + 1 ) * size ], dt / dheight * w );
                atomicAdd( &tmpWater[ x2 + 1 + ( y2 + 1 ) * size ], dtr / dheight * w );
                atomicAdd( &tmpWater[ x2 - 1 + y2 * size ], dml / dheight * w );
                atomicAdd( &tmpWater[ x2 + 1 + y2 * size ], dmr / dheight * w );
                atomicAdd( &tmpWater[ x2 - 1 + ( y2 - 1 ) * size ], dbl / dheight * w );
                atomicAdd( &tmpWater[ x2 + ( y2 - 1 ) * size ], db / dheight * w );
                atomicAdd( &tmpWater[ x2 + 1 + ( y2 - 1 ) * size ], dbr / dheight * w );

                water[ indexGlobal ] = 1.0f + remainingWater;
            }

            dheightV[ indexGlobal ] = dheight;
    //    }
    //}
}

__global__
void erodeSecondPassKernel( float* __restrict__ heightmap, int size, float* __restrict__ water, float* __restrict__ tmpWater, float* __restrict__ dheightV )
{
    const float Scale = 1.0f;
    const float Erosion = 0.0005 * Scale;
    const float Deposition = 0.0000002 * Scale;

    int startx = threadIdx.x + blockIdx.x * blockDim.x;
    int starty = threadIdx.y + blockIdx.y * blockDim.y;
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;

    for( int y = 1 + starty; y < size - 1; y += stridey )
    {
        for( int x = 1 + startx; x < size - 1; x += stridex )
        {
            int index = x + y * size;

            water[ index ] += tmpWater[ index ];
            tmpWater[ index ] = 0;

            float oldHeight = heightmap[ index ];
            heightmap[ index ] += ( -( dheightV[ index ] - 0.005f / Scale ) * water[ index ] ) * Erosion + water[ index ] * Deposition;

            if( oldHeight < heightmap[ index ] )
                water[ index ] = max( water[ index ] - ( heightmap[ index ] - oldHeight ) * 1000.0f, 0.0f );
        }
    }
}

void gpuErosion( int size, int iterations, bool multiPass )
{
    int bufferSize = sizeof( float ) * size * size;

    float* heightmap;
    cudaMallocManaged( &heightmap, bufferSize );

    float* water;
    cudaMallocManaged( &water, bufferSize );

    float* tmpWater;
    cudaMallocManaged( &tmpWater, bufferSize );

    float* dheightV;
    cudaMallocManaged( &dheightV, bufferSize );

    auto noise = generateNoiseTexture( size );

    for( unsigned int i = 0; i < noise.size(); i++ )
        heightmap[ i ] = noise[ i ];

    auto start = std::chrono::high_resolution_clock::now();

    if( multiPass )
    {
        for( int i = 0; i < iterations; i++ )
        {
            erodeFirstPassKernel<<< dim3( 64, 64 ), dim3( 32, 32 ) >>>( heightmap, size, water, tmpWater, dheightV );
            erodeSecondPassKernel<<< 4, dim3( 32, 32 ) >>>( heightmap, size, water, tmpWater, dheightV );
        }
    }
    else
    {
        erodeKernel<<< 4, dim3( 32, 32 ) >>>( heightmap, size, iterations, water, tmpWater, dheightV );
    }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast< std::chrono::milliseconds >( end - start ).count();

    std::cout << "gpu erosion took " << duration << "ms" << std::endl;
    save( "noiseGPU.raw", size, size, std::vector< float >( heightmap, heightmap + size * size ) );

    cudaFree( heightmap );
    cudaFree( water );
    cudaFree( tmpWater );
    cudaFree( dheightV );
}

int main()
{
    const int HeightmapSize = 2048;
    const int Iterations = 300;
    //cpuErosion( HeightmapSize, Iterations );
    gpuErosion( HeightmapSize, Iterations, true );
    return 0;
}