# pythonのインストール
Ubuntuにpythonをインストールします。

プログラミングゼミで最終的に作成したい、環境のイメージ図を以下に示します。

![`overview`](../image/00_python_installation/wsl_pyenv_pipenv_overview.png)

## Pythonバージョンとパッケージ
Pythonは以下の2つのバージョンを管理する必要があります。
- Python本体
- 外部パッケージ

そこで、それぞれPython本体のバージョンをpyenv、外部パッケージをpipenvというツールを使って管理することにします。

## Pyenv+Pipenvインストール

### 前準備
そもそもpythonが入ってないのでインストールします。

- python3とpip3をインストール

```
sudo apt install python3 python3-pip -y
```


### pyenv
Python本体の複数バージョンを使い分けるツールです。

- pyenvに必要なパッケージのインストール

pyenvはインストール時にpythonのコンパイルを行っています。ここではコンパイルに必要なパッケージをaptでインストールします。
```
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```
※1行

y はインストール続行確認を自動で許可するオプション

- pyenvのクローン

```
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```

- pyenv のパスを通す

以下のコマンドを1行ずつ実行します．1番目のコマンドはpyenvが存在するディレクトリの指定，2番目のコマンドはpyenvのバイナリ(実行ファイル)が存在する場所を環境変数に追加しています．
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```

- `~/.bashrc`に追記したので読み込む（`~/.bashrc`とは、shellの起動時に自動で読み込まれる設定ファイル）

```
source ~/.bashrc
```

- 以下コマンドでちゃんと動くか（パスが通ってるか）確認

```
pyenv
```

### pipenv
Pipenvは、Python公式が正式に推薦する依存関係管理・パッケージングのためのツールです。

- 以下pip3でインストール

```
sudo pip3 install pipenv
```

- プロジェクトディレクトリ配下に仮想環境を作って欲しいので、`~/.bashrc`に以下を追記

```
export PIPENV_VENV_IN_PROJECT=true
```

- `~/.bashrc`に追記したので読み込む

```
source ~/.bashrc
```

#### 使い方

ググればわかりやすい記事たくさんでてくるので、Qiita記事コピペしました。

```
pipenv install
# ディレクトリ内にPipfile Pipfile.lockを作成
# 既にある場合はPipfileを元に仮想環境が構築される

pipenv install [package]
# 仮想環境にライブラリをインストールする
# インストール後、Pipfileに追加される
# version指定も可能

pipenv uninstall [package]
# 仮想環境のライブラリ削除
# Pipfileからも削除される
# ※依存ライブラリは削除されない

pipenv shell
# 仮想環境を有効化する

pipenv run [コマンド]
# 仮想環境でコマンドを実行

pipenv --rm
# 仮想環境を削除

pipenv --venv
# 仮想環境のパスを確認できる
```



## Pyenv+Pipenvで仮想環境作成

pipenvが有効化する範囲は各ディレクトリごとです。
今回はB3seminarディレクトリを作ってそこで使いましょう。

- プロジェクトディレクトリ作成

```
mkdir B3Seminar_yourname
cd B3Seminar_yourname
```

- Python3.8の仮想環境作成

```
pipenv --python 3.8.10
```
もしWould you like us to install...[Y/n]って聞かれたらEnter

- 仮想環境の有効化

```
pipenv shell
```

- Pythonバージョン確認

```
python -V
```
Python 3.8.x であることを確認(xは任意の数字)

- ライブラリインストール

試しにnumpyと呼ばれる数値計算ライブラリを入れてみましょう
```
pipenv install numpy
```

- 3x3の行列積してみる

以下の実行例は、要素が全部1の行列aと、要素が全部3の行列bの行列積を計算しています。

```python
$ python
Python 3.7.7 (default, Dec 21 2020, 21:09:12)
[Clang 11.0.3 (clang-1103.0.32.29)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> a = np.ones((3,3))
>>> b = np.ones((3,3)) * 3
>>> a.dot(b)
array([[9., 9., 9.],
       [9., 9., 9.],
       [9., 9., 9.]])
```
