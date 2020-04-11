//https://www.shadertoy.com/view/tdjczh
Shader "Unlit/JuliaGlass"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
		iChannel0("Texture Cube 090 3309 7306", CUBE) = "white" {}
		_MapC("Map Value", Vector) = (0.5, 0.5, 0.5, 0.5)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

#if HW_PERFORMANCE==0
#else
#define AA
#endif

#define BIASED_NORMAL 0
#define MAX_BOUNCES 4
#define MAX_DIST 8.
#define ABSORB  float3(.2, 1., .8) //float3(0., 0.5, 1.3)

			static const int maxIterations = 6;
			float4 _MapC;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

			// fragment shader
			samplerCUBE iChannel0;

			float sdSphere(float3 p, float s)
			{
				return length(p) - s;
			}

			float saturate(float c)
			{
				return clamp(c, 0., 1.);
			}

			float4 quatMult(float4 q1, float4 q2)
			{
				float4 r;
				r.x = q1.x*q2.x - dot(q1.yzw, q2.yzw);
				r.yzw = q1.x*q2.yzw + q2.x*q1.yzw + cross(q1.yzw, q2.yzw);
				return r;
			}

			float4 quatSq(float4 q)
			{
				float4 r;
				//r.x = q.x*q.x - dot(q.yzw, q.yzw);
				//r.yzw = 2.*q.x*q.yzw;

				r.y = q.y*q.y - dot(q.xzw, q.xzw);
				r.xzw = 2.*q.y*q.xzw;
				return r;
			}

#define ESCAPE_THRESHOLD 1e1

			void iterateIntersect(inout float4 q, inout float4 qp, float4 c, int maxIterations)
			{
				for (int i = 0; i < maxIterations; i++)
				{
					qp = 2.0 * quatMult(q, qp);
					q = quatSq(q) + c;
					if (dot(q, q) > ESCAPE_THRESHOLD)
					{
						break;
					}
				}
			}

			float map(in float3 p)
			{
				float4 z = float4(p, 0);
				float4 zp = float4(1., 0, 0, 0);
				float t = _Time.y * 0.2;// 0.1;
				//float4 c = 0.5*float4(cos(t), cos(t*1.1), cos(t*2.3), cos(t*3.1));
				float4 c = _MapC;// .6*float4(cos(t), cos(t*1.1), cos(t*2.3), cos(t*3.1));
				iterateIntersect(z, zp, c, 4);// maxIterations);
				float normZ = length(z);
				float d = 0.5 * normZ * log(normZ) / length(zp);
				return d;
			}

			float3 getSkyColor(float3 rd)
			{
				float3 col = texCUBE(iChannel0, rd).rgb;
				return col * col;
			}

			float rayMarch(in float sgn, in float3 ro, in float3 rd, in float offT)
			{
				float t = offT;
				for (int i = 0; i < 240; i++)
				{
					float h = sgn * map(ro + rd * t);
					t += h;
					if (h < 0.001 || t > MAX_DIST)
					{
						break;
					}
				}
				return t;
			}

#define EPS 0.02
#if BIASED_NORMAL
			float3 calcNormal(float3 pos)
			{
				float ref;
				float trans;
				float3 absorb;
				float3 col;
				float2 eps = float2(EPS, 0);
				float d = map(pos);
				return normalize(float3(map(pos + eps.xyy) - d, map(pos + eps.yxy) - d, map(pos + eps.yyx) - d));
			}
#else
			float3 calcNormal(in float3 pos)
			{
				static const float ep = EPS;
				float2 e = float2(1.0, -1.0)*0.5773;
				return normalize(
					e.xyy*map(pos + e.xyy*ep) +
					e.yyx*map(pos + e.yyx*ep) +
					e.yxy*map(pos + e.yxy*ep) +
					e.xxx*map(pos + e.xxx*ep)
				);
			}
#endif

			float3 Render(in float3 ro, in float3 rd, in float2 uv)
			{
				float sgn = 1.;
				float cref = 0.7;
				float3 col = float3(0, 0, 0);
				float3 rel = float3(1, 1, 1);
				float transp = 1.;
				float3 absorb = float3(uv.y*9.0, uv.x*5., uv.y*2.0)*0.3;// ABSORB * sin(_Time.y);
				//float3 absorb = float3(uv.x, 0.5, uv.y);
				for (int i = 0; i < MAX_BOUNCES; i++)
				{
					float t = rayMarch(sgn, ro, rd, 0.02);
					if (t > MAX_DIST)
					{
						col += rel * getSkyColor(rd)*0.1;
						break;
					}
					float3 rabs = lerp(absorb, float3(0, 0, 0), (sgn + 1.) / 2.);
					float3 beerlamb = exp(-rabs);
					float3 p = ro + rd * t;
					float3 n = sgn * calcNormal(p);
					float3 refl = reflect(rd, n);
					float3 refr = refract(rd, n, cref);

					float fresnel = pow(1.0 - abs(dot(n, rd)), 2.0);
					float reflectorFactor = lerp(0.2, 1.0, fresnel);
					float refractionFactor = lerp(transp, 0., fresnel);

					//col += (1.0 - refractionFactor) * rel * beerlamb * getSkyColor(refl) * reflectorFactor;
					col += (1.0 - refractionFactor) * rel * beerlamb * getSkyColor(refl) * reflectorFactor;
					rel *= refractionFactor *beerlamb;

					ro = p;
					if (refr.x == 0.0 && refr.y == 0.0 && refr.z == 0.0)//(refr == float3(0.0, 0.0, 0.0))
					{
						rd = refl;
					}
					else
					{
						rd = refr;
						sgn *= -1.;
						cref = 1. / cref;
					}
				}

				col += rel * getSkyColor(rd);
				return col;
			}

			float3 contrast(in float3 color, in float c)
			{
				float t = 0.5 - c * 0.5;
				return color * c + t;
			}

			float3 vignette(float3 color, float2 q, float v)
			{
				color *= 0.3 + 0.8 * pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), v);
				return color;
			}

			float3 postProcess(in float3 col, in float2 q)
			{
				col = pow(col, float3(0.8, 0.8, 0.8));
				col = contrast(col, 1.2);
				col = vignette(col, q, 0.8);
				return col;
			}

			float3x3 setCamera(in float3 ro, in float3 ta)
			{
				float3 cw = normalize(ta - ro);
				float3 up = float3(0, 1, 0);
				float3 cu = normalize(cross(cw, up));
				float3 cv = normalize(cross(cu, cw));
				return float3x3(cu, cv, cw);
			}

			float3x3 genMat(in float3 dir, in float3 up)
			{
				float3 cw = normalize(dir);
				float3 cu = normalize(cross(cw, up));
				float3 cv = normalize(cross(cu, cw));
				return float3x3(cu, cv, cw);
			}

            fixed4 frag (v2f i) : SV_Target
            {
				float3 tot = float3(0.0, 0.0, 0.0);
				float2 rook[4];
				//rook[0] = float2(1.*0.125, 3.*0.125);// (1. / 8., 3. / 8.);
				//rook[1] = float2(3.*0.125, -1.*0.125);// (3. / 8., -1. / 8.);
				/*rook[2] = float2(-1. / 8., -3. / 8.);
				rook[3] = float2(-3. / 8., 1. / 8.);*/

				for (int n = 0; n < 1; n++)
				{
					// pixel coordinates
					//float2 o = rook[n];
					//float2 p = (-_ScreenParams.xy + 2.0*(i.uv.xy*_ScreenParams.xy + o)) / _ScreenParams.y;
					float2 p = (-_ScreenParams.xy + 2.0*(i.uv.xy*_ScreenParams.xy)) / _ScreenParams.y;

					float theta = 0.; _Time.y * 0.1;// radians(360.)*
					float phi = 1.2;// radians
					float3 ro = /*2.5*/2.9*float3(sin(phi)*cos(theta), cos(phi), sin(phi)*sin(theta));
					float3 ta = float3(0, 0, 0);

					// camera-to-world transformation
					float3x3 ca = setCamera(ro, ta);
					float3 rd = mul(normalize(float3(p, 1.5)), ca);
					
					float3 col = Render(ro, rd, i.uv.xy);

					tot += col;
				}

				tot *= 0.75;

				tot = postProcess(tot, i.uv.xy);
				return float4(sqrt(tot), 1.0);
            }
            ENDCG
        }
    }
}
