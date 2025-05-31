# ragdoll
trying to see if rag works better with a more creative interface, a tiny tiny embedding model, and 'old' ml tricks like knn

# 3rd party packages
relies heavily on some fantastic projects (please refer to their licenses for usage):
- [The Minish Lab's](https://github.com/MinishLab) [vicinity](https://github.com/MinishLab/vicinity): super lightweight low-dependency vector store for nearest neighbor search, with support for different backends and evaluation
- [The Minish Lab's](https://github.com/MinishLab) [model2vec](https://github.com/MinishLab/model2vec): turn any sentence transformer model into a really small static model - this quick POC uses an 8M parameter model, `minishlab/potion-base-8M`, distilled by the `model2vec` team, from `baai/bge-base-en-v1.5`. crazy fast (as is the 32M). for best performance, you really should train it within your domain. check out their work though - they have some innovative approaches that i haven't come across much before.
- text chunking via [semchunk](https://github.com/isaacus-dev/semchunk), and an embedding based one, [chonkie](https://github.com/chonkie-inc/chonkie)
