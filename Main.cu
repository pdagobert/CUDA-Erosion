#include <vector>
#include <string>
#include <fstream>
#include <iostream>
//#include <chrono>

#include "noise.inl.cu"

__global__
void generateNoiseTextureKernel( float* heightmap, int size )
{
    int startx = threadIdx.x + blockIdx.x * blockDim.x;
    int starty = threadIdx.y + blockIdx.y * blockDim.y;
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;

    for( int y = starty; y < size; y += stridey )
    {
        for( int x = startx; x < size; x += stridex )
        {
            float xpos = (float)x / size * 2.0f - 1.0f;
            float ypos = (float)y / size * 2.0f - 1.0f;

            float factor = 1.8f;
            xpos *= factor;
            ypos *= factor;

            float height = Cuda::fbm( xpos, ypos, 8, 2.0f, 0.5f );
            float dist = sqrt( xpos * xpos + ypos * ypos );
            float alpha = max( 1.0f - dist * dist, 0.0f );

            heightmap[ x + y * size ] = 1.0f * height * 0.5 + 0.5;
        }
    }
}
std::vector< float > generateNoiseTexture( int size )
{
    std::vector< float > heightmap;
    heightmap.reserve( size * size );

    for( int y = 0; y < size; y++ )
    {
        for( int x = 0; x < size; x++ )
        {
            float xpos = (float)x / size * 2.0f - 1.0f;
            float ypos = (float)y / size * 2.0f - 1.0f;

            float factor = 1.8f;
            xpos *= factor;
            ypos *= factor;

            float height = fbm( xpos, ypos, 8, 2.0f, 0.5f );
            float dist = sqrt( xpos * xpos + ypos * ypos );
            float alpha = max( 1.0f - dist * dist, 0.0f );

            heightmap.push_back( 1.0f * ( height * 0.5 + 0.5 ) );
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
                dtl = max( m - dtl, 0.0f );
                float dt = heightmap[ x + ( y + 1 ) * size ];
                dt = max( m - dt, 0.0f );
                float dtr = heightmap[ x + 1 + ( y + 1 ) * size ];
                dtr = max( m - dtr, 0.0f );
                float dml = heightmap[ x - 1 + y * size ];
                dml = max( m - dml, 0.0f );
                float dmr = heightmap[ x + 1 + y * size ];
                dmr = max( m - dmr, 0.0f );
                float dbl = heightmap[ x - 1 + ( y - 1 ) * size ];
                dbl = max( m - dbl, 0.0f );
                float db = heightmap[ x + ( y - 1 ) * size ];
                db = max( m - db, 0.0f );
                float dbr = heightmap[ x + 1 + ( y - 1 ) * size ];
                dbr = max( m - dbr, 0.0f );

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
                    water[ index ] = max( water[index ] - ( heightmap[ index ] - oldHeight ) * 1000.0f, 0.0f );
            }
        }
    }
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
                dtl = max( m - dtl, 0.0f );
                float dt = heightmap[ x + ( y + 1 ) * size ];
                dt = max( m - dt, 0.0f );
                float dtr = heightmap[ x + 1 + ( y + 1 ) * size ];
                dtr = max( m - dtr, 0.0f );
                float dml = heightmap[ x - 1 + y * size ];
                dml = max( m - dml, 0.0f );
                float dmr = heightmap[ x + 1 + y * size ];
                dmr = max( m - dmr, 0.0f );
                float dbl = heightmap[ x - 1 + ( y - 1 ) * size ];
                dbl = max( m - dbl, 0.0f );
                float db = heightmap[ x + ( y - 1 ) * size ];
                db = max( m - db, 0.0f );
                float dbr = heightmap[ x + 1 + ( y - 1 ) * size ];
                dbr = max( m - dbr, 0.0f );

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
                    water[ index ] = max( water[index ] - ( heightmap[ index ] - oldHeight ) * 1000.0f, 0.0f );
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

            int offset = 0;

            if( threadIdx.x == 0 && x2 != 0 )
                offset = -1;

            if( threadIdx.x == 31 && x2 != size - 1 )
                offset = 1;

            if( offset != 0 )
            {
                int borderGlobal = x2 + offset + y2 * size;
                int borderLocal = x + offset + y * tileSize;
                heightmapCache[ borderLocal ] = heightmap[ borderGlobal ];
            }

            offset = 0;

            if( threadIdx.y == 0 && y2 != 0 )
                offset = -1;

            if( threadIdx.y == 31 && y2 != size - 1 )
                offset = 1;

            if( offset != 0 )
            {
                int borderGlobal = x2 + ( y2 + offset ) * size;
                int borderLocal = x + ( y + offset ) * tileSize;
                heightmapCache[ borderLocal ] = heightmap[ borderGlobal ];
            }

            int xoffset = 0;
            int yoffset = 0;

            if( threadIdx.x == 0 && threadIdx.y == 0 && ( x2 != 0 && y2 != 0 ) )
            {
                xoffset = -1;
                yoffset = -1;
            }

            if( threadIdx.x == 31 && threadIdx.y == 0 && ( x2 != size - 1 && y2 != 0 ) )
            {
                xoffset = 1;
                yoffset = -1;
            }

            if( threadIdx.x == 0 && threadIdx.y == 31 && ( x2 != 0 && y2 != size - 1 ) )
            {
                xoffset = -1;
                yoffset = 1;
            }

            if( threadIdx.x == 31 && threadIdx.y == 31 && ( x2 != size - 1 && y2 != size - 1 ) )
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

            if( x2 == 0 || y2 == 0 || x2 == size - 1 || y2 == size - 1 )
                return;

            float dtl = heightmapCache[ x - 1 + ( y + 1 ) * tileSize ];
            float dt = heightmapCache[ x + ( y + 1 ) * tileSize ];
            float dtr = heightmapCache[ x + 1 + ( y + 1 ) * tileSize ];
            float dml = heightmapCache[ x - 1 + y * tileSize ];
            float m = heightmapCache[ indexLocal ];
            float dmr = heightmapCache[ x + 1 + y * tileSize ];
            float dbl = heightmapCache[ x - 1 + ( y - 1 ) * tileSize ];
            float db = heightmapCache[ x + ( y - 1 ) * tileSize ];
            float dbr = heightmapCache[ x + 1 + ( y - 1 ) * tileSize ];

            dtl = max( m - dtl, 0.0f );
            dt = max( m - dt, 0.0f );
            dtr = max( m - dtr, 0.0f );
            dml = max( m - dml, 0.0f );
            dmr = max( m - dmr, 0.0f );
            dbl = max( m - dbl, 0.0f );
            db = max( m - db, 0.0f );
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

void save( const std::string& fileName, int width, int height, const std::vector< float >& heightmap )
{
    std::ofstream file( fileName.c_str(), std::ios::binary );
    if( file.fail() )
    {
        std::cout << "cannot save file " << fileName << std::endl;
        return;
    }

    file.write( reinterpret_cast< char* >( &width ), sizeof( int ) );
    file.write( reinterpret_cast< char* >( &height ), sizeof( int ) );
    file.write( reinterpret_cast< const char* >( &heightmap[ 0 ] ), heightmap.size() * sizeof( float ) );
}

double elapsedTime( clock_t begin, clock_t end )
{
    return (double)( end - begin ) / CLOCKS_PER_SEC * 1000.0;
}

void cpuErosion( int size, int iterations, const std::string& fileName )
{
    clock_t begin = clock();
    std::vector< float > heightmap = generateNoiseTexture( size );
    clock_t end = clock();

    std::cout << "Noise took " << elapsedTime( begin, end ) << " ms" << std::endl;

    if( iterations > 0 )
    {
        begin = clock();
        erode( heightmap, size, iterations );
        end = clock();

        std::cout << "Erosion took " << elapsedTime( begin, end ) << " ms" << std::endl;
    }

    save( fileName, size, size, heightmap );
}

void gpuErosion( int size, int iterations, const std::string& fileName, bool multiPass )
{
    int bufferSize = sizeof( float ) * size * size;

    float* heightmap;
    cudaMallocManaged( &heightmap, bufferSize );

    cudaEvent_t begin, end;
    cudaEventCreate( &begin );
    cudaEventCreate( &end );

    cudaEventRecord( begin );
    generateNoiseTextureKernel<<< 16, 1024 >>>( heightmap, size );
    cudaEventRecord( end );
    cudaEventSynchronize( end );

    float ms;
    cudaEventElapsedTime( &ms, begin, end );

    std::cout << "Noise took " << ms << " ms" << std::endl;

    if( iterations > 0 )
    {
        float* water;
        cudaMallocManaged( &water, bufferSize );

        float* tmpWater;
        cudaMallocManaged( &tmpWater, bufferSize );

        float* dheightV;
        cudaMallocManaged( &dheightV, bufferSize );

        cudaEventRecord( begin );

        if( multiPass )
        {
            const int TileSize = 32;
            dim3 numThreads( TileSize, TileSize );
            dim3 numBlocks( size / TileSize, size / TileSize );

            for( int i = 0; i < iterations; i++ )
            {
                erodeFirstPassKernel<<< numBlocks, numThreads >>>( heightmap, size, water, tmpWater, dheightV );
                erodeSecondPassKernel<<< numBlocks, numThreads >>>( heightmap, size, water, tmpWater, dheightV );
            }
        }
        else
        {
            erodeKernel<<< 4, dim3( 32, 32 ) >>>( heightmap, size, iterations, water, tmpWater, dheightV );
        }

        cudaEventRecord( end );
        cudaEventSynchronize( end );

        cudaEventElapsedTime( &ms, begin, end );

        std::cout << "Erosion took " << ms << " ms" << std::endl;

        cudaFree( water );
        cudaFree( tmpWater );
        cudaFree( dheightV );
    }

    save( fileName, size, size, std::vector< float >( heightmap, heightmap + size * size ) );

    cudaFree( heightmap );
}

int main( int argc, char* argv[] )
{
    if( argc < 4 )
    {
        std::cout << "usage : erosion SIZE ITERATION FILE_NAME [RUN_ON_GPU]" << std::endl;
        std::cout << "usage: SIZE: size of heightmap" << std::endl;
        std::cout << "usage: ITERATION: > 0 for erosion, 0 for noise only" << std::endl;
        std::cout << "usage: FILE_NAME: output file name" << std::endl;
        std::cout << "usage: RUN_ON_GPU: 1 for GPU, 0 for CPU, default to 1" << std::endl;

        return 0;
    }

    int size = atoi( argv[ 1 ] );
    int iterations = atoi( argv[ 2 ] );
    std::string fileName = argv[ 3 ];
    bool gpu = argc == 5 ? atoi( argv[ 4 ] ) : true;

    if( gpu )
        gpuErosion( size, iterations, fileName, true );
    else
        cpuErosion( size, iterations, fileName );

    return 0;
}