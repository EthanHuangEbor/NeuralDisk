# 编译说明

1. 将 `FRB-main_normed.tex` 复制到你原来的 `src/` 模板目录下，与 `FRB-style.cls`、`fonts/`、`img/` 同级。
2. 将本包中的 `figures/` 目录也复制到 `src/` 目录下。
3. 使用 XeLaTeX 编译，建议连续编译两次以生成目录和引用编号：

```bash
xelatex FRB-main_normed.tex
xelatex FRB-main_normed.tex
```

4. 本包未包含字体文件。字体沿用你已有模板 `src/fonts/` 中的文件。
5. 正文中已用当前阶段真实进度替换旧版“示例值”；三维有限元再分析和神经网络训练结果仍保留为待完成框架，不填充虚假数值。
