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
    auto heightmap = generateNoiseTexture( 1024 );
    save( "noise.raw", 1024, 1024, heightmap );
    return 0;
}