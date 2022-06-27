# Glasses 😎

Compact, concise and customizable deep learning computer vision library

## TL;DR

This library has

- human readable code, no *research code*
- common component are shared across [models](#Models)
- [same APIs](#classification) for all models (you learn them once and they are always the same)
- clear and easy to use model constomization (see [here](#block))
- [classification](#classification) and [segmentation](#segmentation) 
- emoji in the name ;)

## Getting Started

Head over the [getting started](getting_started) guide

## Cool stuff

### Code Highlights

See [here](https://squidfunk.github.io/mkdocs-material/reference/code-blocks/) for doc

```python

print("Hello World!")
```

### Admonition

See [here](https://squidfunk.github.io/mkdocs-material/reference/admonitions/) for doc

!!! note "Hello There"
    General Kenobi!

??? note "Hello There"
    General Kenobi!

!!! error
    Boom!


### Latex

See [here](https://squidfunk.github.io/mkdocs-material/reference/mathjax/) for doc

Inline

$p(x|y) = \frac{p(y|x)p(x)}{p(y)}$, \(p(x|y) = \frac{p(y|x)p(x)}{p(y)}\).

Block

$$
E(\mathbf{v}, \mathbf{h}) = -\sum_{i,j}w_{ij}v_i h_j - \sum_i b_i v_i - \sum_j c_j h_j
$$

\begin{align}
    p(v_i=1|\mathbf{h}) & = \sigma\left(\sum_j w_{ij}h_j + b_i\right) \\
    p(h_j=1|\mathbf{v}) & = \sigma\left(\sum_i w_{ij}v_i + c_j\right)
\end{align}