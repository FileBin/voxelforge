/* Тексты шейдеров в одном месте: их правят чаще всего, и искать их по коду
 * рендерера неудобно. */

export const VS_BODY = `#version 300 es
precision highp float;
in vec3 aPos;
in vec3 aNrm;
in vec3 iPos;
in vec3 iScale;
uniform mat4 uVP;
out vec3 vN;
out vec3 vW;
void main() {
  vW = aPos * iScale + iPos;
  vN = aNrm;                     /* оси-выровненные коробки: масштаб не меняет нормаль */
  gl_Position = uVP * vec4(vW, 1.0);
}`;

export const FS_BODY = `#version 300 es
precision highp float;
in vec3 vN;
in vec3 vW;
uniform vec3 uEye;
uniform vec3 uTint;
out vec4 outColor;
void main() {
  vec3 n = normalize(vN);
  vec3 L = normalize(vec3(0.45, 0.9, 0.35));
  float diff = max(dot(n, L), 0.0);
  float sky = 0.5 + 0.5 * n.y;
  vec3 col = uTint * (0.22 + 0.5 * sky) + uTint * diff * 0.85;
  vec3 V = normalize(uEye - vW);
  vec3 H = normalize(L + V);
  col += vec3(1.0) * pow(max(dot(n, H), 0.0), 48.0) * 0.22;
  float fog = clamp(1.0 - exp(-length(uEye - vW) * 0.012), 0.0, 0.8);
  col = mix(col, vec3(0.043, 0.059, 0.078), fog);
  outColor = vec4(col, 1.0);
}`;

/* Меш воксельного тела: форма хранится в локальных координатах, кадр задаёт
 * жёсткое движение — поворот кватернионом и перенос. */
export const VS_WORLD = `#version 300 es
precision highp float;
in vec3 aPos;
in vec3 aNrm;
uniform mat4 uVP;
uniform vec3 uPos;
uniform vec4 uQuat;
out vec3 vN;
out vec3 vW;
vec3 qrot(vec4 q, vec3 v) {
  vec3 t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
void main() {
  vec3 p = qrot(uQuat, aPos) + uPos;
  vW = p;
  vN = qrot(uQuat, aNrm);
  gl_Position = uVP * vec4(p, 1.0);
}`;

export const FS_WORLD = `#version 300 es
precision highp float;
in vec3 vN;
in vec3 vW;
uniform vec3 uEye;
uniform vec3 uTint;
out vec4 outColor;
void main() {
  vec3 n = normalize(vN);
  vec3 L = normalize(vec3(0.45, 0.9, 0.35));
  float diff = max(dot(n, L), 0.0);
  float sky = 0.5 + 0.5 * n.y;
  vec3 col = uTint * (0.20 + 0.42 * sky) + uTint * diff * 0.9;
  vec3 V = normalize(uEye - vW);
  vec3 H = normalize(L + V);
  col += vec3(1.0) * pow(max(dot(n, H), 0.0), 56.0) * 0.20;
  float fog = clamp(1.0 - exp(-length(uEye - vW) * 0.010), 0.0, 0.8);
  col = mix(col, vec3(0.043, 0.059, 0.078), fog);
  outColor = vec4(col, 1.0);
}`;

export const VS_GRID = `#version 300 es
precision highp float;
in vec3 aPos;
uniform mat4 uVP;
out vec2 vXZ;
void main() {
  vXZ = aPos.xz;
  gl_Position = uVP * vec4(aPos, 1.0);
}`;

export const FS_GRID = `#version 300 es
precision highp float;
in vec2 vXZ;
uniform vec3 uEye;
out vec4 outColor;
float lineAt(vec2 p, float stepSize) {
  vec2 g = abs(fract(p / stepSize - 0.5) - 0.5) / fwidth(p / stepSize);
  return 1.0 - min(min(g.x, g.y), 1.0);
}
void main() {
  float fine = lineAt(vXZ, 0.5) * 0.35;
  float big = lineAt(vXZ, 2.0) * 0.55;
  float a = clamp(fine + big, 0.0, 1.0);
  vec3 col = mix(vec3(0.043, 0.059, 0.078), vec3(0.22, 0.29, 0.36), a);
  float fog = clamp(1.0 - exp(-length(uEye - vec3(vXZ.x, 0.0, vXZ.y)) * 0.012), 0.0, 0.85);
  col = mix(col, vec3(0.043, 0.059, 0.078), fog);
  outColor = vec4(col, 1.0);
}`;

export const VS_MESH = `#version 300 es
precision highp float;
in vec3 aPos;
uniform mat4 uVP;
out vec3 vW;
void main() {
  vW = aPos;
  gl_Position = uVP * vec4(aPos, 1.0);
}`;

/* Нормали считаем производными: в .vfm их нет, а для воксельной поверхности
 * фасеточный вид даже правильнее — видно, из каких плоскостей собрана форма. */
export const FS_MESH = `#version 300 es
precision highp float;
in vec3 vW;
uniform vec3 uEye;
uniform vec3 uTint;
out vec4 outColor;
void main() {
  vec3 n = normalize(cross(dFdx(vW), dFdy(vW)));
  if (dot(n, uEye - vW) < 0.0) n = -n;
  vec3 L = normalize(vec3(0.45, 0.9, 0.35));
  float diff = max(dot(n, L), 0.0);
  float sky = 0.5 + 0.5 * n.y;
  vec3 col = uTint * (0.2 + 0.45 * sky) + uTint * diff * 0.9;
  vec3 H = normalize(L + normalize(uEye - vW));
  col += vec3(1.0) * pow(max(dot(n, H), 0.0), 40.0) * 0.2;
  outColor = vec4(col, 1.0);
}`;

/* Линии сетки вокселей: те же перенос и поворот, что у меша. */
export const VS_GRIDLINES = `#version 300 es
precision highp float;
in vec3 aPos;
uniform mat4 uVP;
uniform vec3 uPos;
uniform vec4 uQuat;
vec3 qrot(vec4 q, vec3 v) {
  vec3 t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
void main() {
  gl_Position = uVP * vec4(qrot(uQuat, aPos) + uPos, 1.0);
}`;

export const VS_LINE = `#version 300 es
precision highp float;
in vec3 aPos;
uniform mat4 uVP;
void main() { gl_Position = uVP * vec4(aPos, 1.0); }`;

export const FS_LINE = `#version 300 es
precision highp float;
uniform vec3 uTint;
out vec4 outColor;
void main() { outColor = vec4(uTint, 0.55); }`;
