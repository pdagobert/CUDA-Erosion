#include <vector>
#include <string>
#include <fstream>
#include <iostream>

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
void erode( std::vector< float >& heightmap, int size )
{
    const float Scale = 1.0f;
    const float Erosion = 0.0005 * Scale;
    const float Deposition = 0.0000002 * Scale;
    const float Evaporation = 0.9f;
    const int iterations = 300;

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
                float tl = heightmap[ x - 1 + ( y + 1 ) * size ];
                float t = heightmap[ x + ( y + 1 ) * size ];
                float tr = heightmap[ x + 1 + ( y + 1 ) * size ];
                float ml = heightmap[ x - 1 + y * size ];
                float m = heightmap[ x + y * size ];
                float mr = heightmap[ x + 1 + y * size ];
                float bl = heightmap[ x - 1 + ( y - 1 ) * size ];
                float b = heightmap[ x + ( y - 1 ) * size ];
                float br = heightmap[ x + 1 + ( y - 1 ) * size ];

                float dheight = glm::max( m - tl, 0.0f );
                dheight += glm::max( m - t, 0.0f );
                dheight += glm::max( m - tr, 0.0f );
                dheight += glm::max( m - ml, 0.0f );
                dheight += glm::max( m - mr, 0.0f );
                dheight += glm::max( m - bl, 0.0f );
                dheight += glm::max( m - b, 0.0f );
                dheight += glm::max( m - br, 0.0f );

                dheightV[ x + y * size ] = dheight;

                if( dheight != 0.0f )
                {
                    float w = water[ x + y * size ] * Evaporation;
                    float remainingWater = w * 0.0002 / ( dheight * Scale + 1.0f );
                    w -= remainingWater;

                    dWater[ x - 1 + ( y + 1 ) * size ] += ( glm::max( m - tl, 0.0f ) / dheight ) * w;
                    dWater[ x + ( y + 1 ) * size ] += ( glm::max( m - t, 0.0f ) / dheight ) * w;
                    dWater[ x + 1 + ( y + 1 ) * size ] += ( glm::max( m - tr, 0.0f ) / dheight ) * w;
                    dWater[ x - 1 + y * size ] += ( glm::max( m - ml, 0.0f ) / dheight ) * w;
                    dWater[ x + 1 + y * size ] += ( glm::max( m - mr, 0.0f ) / dheight ) * w;
                    dWater[ x - 1 + ( y - 1 ) * size ] += ( glm::max( m - bl, 0.0f ) / dheight ) * w;
                    dWater[ x + ( y - 1 ) * size ] += ( glm::max( m - b, 0.0f ) / dheight ) * w;
                    dWater[ x + 1 + ( y - 1 ) * size ] += ( glm::max( m - br, 0.0f ) / dheight ) * w;

                    water[ x + y * size ] = 1.0f + remainingWater;
                }
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

int main()
{
    const int HeightmapSize = 1024;

    auto heightmap = generateNoiseTexture( HeightmapSize );
    erode( heightmap, HeightmapSize );

    save( "noise.raw", HeightmapSize, HeightmapSize, heightmap );
    return 0;
}