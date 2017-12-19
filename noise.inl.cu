static const unsigned char Permutations[ 512 ] =
{
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

static const float Lut2[ 8 ][ 2 ] =
{
    { -1.0f, -1.0f }, { 1.0f, 0.0f }, { -1.0f, 0.0f }, { 1.0f, 1.0f },
    { -1.0f, 1.0f }, { 0.0f, -1.0f }, { 0.0f, 1.0f }, { 1.0f, -1.0f }
};

void findGradient( int hash, float* gx, float* gy )
{
    int h = hash & 7;
    *gx = Lut2[ h ][ 0 ];
    *gy = Lut2[ h ][ 1 ];
}

float computeSimplexContribution( float x, float y, int i, int j )
{
    float n = 0.0f;

    float t = 0.5f - x * x - y * y;
    if( t >= 0.0f )
    {
        float gx, gy;
        findGradient( Permutations[ i + Permutations[ j ] ], &gx, &gy );

        t = t * t * t * t;
        n = t * ( x * gx + y * gy );
    }

    return n;
}

//https://github.com/stegu/perlin-noise/blob/master/src/sdnoise1234.c
float simplex2D( float x, float y )
{
    const float F = 0.366025403f;
    const float G = 0.211324865f;

    float s = ( x + y ) * F;
    float xs = x + s;
    float ys = y + s;

    int i = floor( xs );
    int j = floor( ys );

    float t = ( i + j ) * G;

    float x0 = x - ( i - t );
    float y0 = y - ( j - t );

    int i1, j1;

    if( x0 > y0 )
    {
        i1 = 1;
        j1 = 0;
    }
    else
    {
        i1 = 0;
        j1 = 1;
    }

    float x1 = x0 - i1 + G;
    float y1 = y0 - j1 + G;
    float x2 = x0 - 1.0f + 2.0f * G;
    float y2 = y0 - 1.0f + 2.0f * G;

    int ii = i & 0xFF;
    int jj = j & 0xFF;

    float n0 = computeSimplexContribution( x0, y0, ii, jj );
    float n1 = computeSimplexContribution( x1, y1, ii + i1, jj + j1 );
    float n2 = computeSimplexContribution( x2, y2, ii + 1, jj + 1 );

    return 40.0f * ( n0 + n1 + n2 );
}

float fbm( float x, float y, int octaves, float lacunarity, float gain )
{
    float value = 0.0f;
    float frequency = 1.0f;
    float amplitude = 0.5f;

    for( int i = 0; i < octaves; i++ )
    {
        float noise = simplex2D( x * frequency, y * frequency );

        value += noise * amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return value;
}

float fbmRidged( float x, float y, int octaves, float lacunarity, float gain )
{
    float value = 0.0f;
    float amplitude = 0.5f;
    float frequency = 1.0f;

    for( int i = 0; i < octaves; i++ )
    {
        float noise = 1.0 - abs( simplex2D( x * frequency, y * frequency ) );

        value += noise * amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }

    return value;
}

namespace Cuda
{
    __constant__
    static const unsigned char Permutations[ 512 ] =
    {
        151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
        190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
        88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
        129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
        49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
        151,160,137,91,90,15,
        131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
        190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
        88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
        77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
        102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
        135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
        5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
        223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
        129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
        251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
        49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
    };

    __constant__
    static const float Lut2[ 8 ][ 2 ] =
    {
        { -1.0f, -1.0f }, { 1.0f, 0.0f }, { -1.0f, 0.0f }, { 1.0f, 1.0f },
        { -1.0f, 1.0f }, { 0.0f, -1.0f }, { 0.0f, 1.0f }, { 1.0f, -1.0f }
    };

    __device__
    void findGradient( int hash, float* gx, float* gy )
    {
        int h = hash & 7;
        *gx = Lut2[ h ][ 0 ];
        *gy = Lut2[ h ][ 1 ];
    }

    __device__
    float computeSimplexContribution( float x, float y, int hash )
    {
        float n = 0.0f;

        float t = 0.5f - x * x - y * y;
        if( t >= 0.0f )
        {
            float gx, gy;
            findGradient( hash, &gx, &gy );

            t = t * t * t * t;
            n = t * ( x * gx + y * gy );
        }

        return n;
    }

    __device__
    //https://github.com/stegu/perlin-noise/blob/master/src/sdnoise1234.c
    float simplex2D( float x, float y )
    {
        const float F = 0.366025403f;
        const float G = 0.211324865f;

        float s = ( x + y ) * F;
        float xs = x + s;
        float ys = y + s;

        int i = floor( xs );
        int j = floor( ys );

        float t = ( i + j ) * G;

        float x0 = x - ( i - t );
        float y0 = y - ( j - t );

        int i1, j1;

        if( x0 > y0 )
        {
            i1 = 1;
            j1 = 0;
        }
        else
        {
            i1 = 0;
            j1 = 1;
        }

        float x1 = x0 - i1 + G;
        float y1 = y0 - j1 + G;
        float x2 = x0 - 1.0f + 2.0f * G;
        float y2 = y0 - 1.0f + 2.0f * G;

        int ii = i & 0xFF;
        int jj = j & 0xFF;

        float n0 = computeSimplexContribution( x0, y0, Permutations[ ii + Permutations[ jj ] ] );
        float n1 = computeSimplexContribution( x1, y1, Permutations[ ii + i1 + Permutations[ jj + j1 ] ] );
        float n2 = computeSimplexContribution( x2, y2, Permutations[ ii + 1 + Permutations[ jj + 1 ] ] );

        return 40.0f * ( n0 + n1 + n2 );
    }

    __device__
    float fbm( float x, float y, int octaves, float lacunarity, float gain )
    {
        float value = 0.0f;
        float frequency = 1.0f;
        float amplitude = 0.5f;

        for( int i = 0; i < octaves; i++ )
        {
            float noise = simplex2D( x * frequency, y * frequency );

            value += noise * amplitude;
            frequency *= lacunarity;
            amplitude *= gain;
        }

        return value;
    }

    __device__
    float fbmRidged( float x, float y, int octaves, float lacunarity, float gain )
    {
        float value = 0.0f;
        float amplitude = 0.5f;
        float frequency = 1.0f;

        for( int i = 0; i < octaves; i++ )
        {
            float noise = 1.0 - abs( simplex2D( x * frequency, y * frequency ) );

            value += noise * amplitude;
            frequency *= lacunarity;
            amplitude *= gain;
        }

        return value;
    }
}