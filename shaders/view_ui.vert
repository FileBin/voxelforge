#version 450
// Оверлей диагностики: вершины уже в NDC (пиксели в NDC считает CPU),
// цвет приходит с вершиной. Ни дескрипторов, ни push-констант: один
// динамический вершинный буфер на кадр.
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;

layout(location = 0) out vec4 vColor;

void main() {
    vColor = aColor;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
