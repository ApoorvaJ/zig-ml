# ⚡ zig-ml

Large Language Model inference written in the [Zig](https://ziglang.org/).

I started this project to understand how AI inference works, without using any high-level libraries. I also want learn how to optimize these workloads and run them on modest hardware.

## Setup

1. Clone this repo
2. [Install Zig](https://github.com/ziglang/zig/wiki/Install-Zig-from-a-Package-Manager)
3. Download the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) model, which is currently the only one tested with this project: \
`wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin`
4. Build and run with a custom prompt: \
`zig build run -- ./stories42M.bin -i "There was a dog called Milo"`

You should get an output that looks something like this:

```
There was a dog called Milo. Milo was a very happy dog. He loved to play and run around. One day, Milo was playing in the park when he saw a big, scary cat. The cat was very mean and it made Milo feel very scared.Milo wanted to run away, but he was too scared. He tried to run away, but the cat was too fast. The cat chased Milo and he ran as fast as he could.Milo ran and ran until he was very tired. He stopped and looked around. He saw a big tree and he knew he was safe. He looked up and saw the cat in the tree.Milo was very scared, but he was also very brave. He decided to try and get the cat down. He jumped up and barked at the cat. The cat was so scared that it ran away.Milo was very happy. He had been so brave and he had been so brave. He ran back home and he was very happy. He had been very brave and he was very proud of himself.
```

Not exactly Shakespeare, but not bad for a relatively small model.

## Future work

This is a part time project that I want to have some fun with. I'd like to follow my curiosity and add various optimizations to run larger models. We'll see where time takes us!

## Contributing

For now, I'd like to keep this a small and personal project, so I won't be accepting any feature PRs. Feel free to fork this repo instead. Please do report any bugs though! :)

## Thanks

1. Andrej Karpathy, for [karpathy/llama2.c](https://github.com/karpathy/llama2.c), which this project is very heavily based on.
2. Jonathan Marler, for his [Zig memory mapping abstraction](https://github.com/marler8997/zigx/blob/14b183c4b0b4e1060ea398f9b05e818ee73f152f/MappedFile.zig), that works on Windows, MacOS, and Linux.
3. To the Zig team for creating a fun programming language.
