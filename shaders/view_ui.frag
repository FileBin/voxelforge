#version 450
layout(location = 0) in vec4 vColor;

layout(location = 0) out vec4 outColor;

void main() {
    // Цвет уже с альфой: панель полупрозрачная, текст и полосы — плотные.
    outColor = vColor;
}
