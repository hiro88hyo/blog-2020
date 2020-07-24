---
template: post
title: 機械学習の分類
slug: categories-of-machine-learning
draft: false
date: 2020-02-10T22:45:31.204Z
description: 機械学習を勉強するにあたって、どういう順番で手をつけていこうか。これは自分にとっても非常に悩ましかった。なにしろ基礎知識すらあまりなかったわけで、いろいろあさっていくなかで、なんとなく方向性みたいなのが見えてきたのでまとめてみることにした。万人に参考になるかはなぞである。
category: programming
tags:
  - 機械学習
  - データサイエンス
  - Python
  - scikit-learn
---
# はじめに
機械学習を勉強するにあたって、どういう順番で手をつけていこうか。これは自分にとっても非常に悩ましかった。なにしろ基礎知識すらあまりなかったわけで、いろいろあさっていくなかで、なんとなく方向性みたいなのが見えてきたのでまとめてみることにした。万人に参考になるかはなぞである。

# ライブラリあるんだから使えばいいじゃん
それは実に正しい。エンジンの仕組みなんて知らなくても車の運転はできる。大事なのは車を使ってどういう価値を生み出すかだ、車輪の再発明をやっている時間はないのだよ。機械学習ならscikit-learn、ディープラーニングをやるならtensorflowを使えばナウでヤングなAIなんてあっという間ですよ！

確かにそうですけどね、ベースにある理論とか知識とかがあったほうがより適切かつ効果的に道具を選択できるようになると思うんです。あーこのケースならこういう風に解いていけばいいなというのが感覚的にわかるようになるのはとても大事だと思います。

# ではどのようにして
結局行き着いたのは、[scikit-learnのチートシート](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)と[Microsoftのチートシート](https://docs.microsoft.com/ja-jp/azure/machine-learning/algorithm-cheat-sheet)。機械学習の分類についてよくまとまっているように感じた。scikit-learnは若干範囲が狭く、より広い分野、新しい分野を扱うならMicrosoftの方がいいと思う。

今後は、以下にあげる項目についての理論的な話からライブラリに頼らずにpython実装を考えてみる。それからライブラリを使えるように進めていきたいと思う。すでに似たようなもっと良質な記事がたくさんあるので、そちらに譲ることも多いと思う（手抜き宣言）。

理解するにあたって必要になってくる高校レベルの数学（微分積分とか行列とか確率統計とか）や、pythonの基本的な使い方については割愛する予定である。

# まず機械学習の分類
カテゴリとしては大きくは3つに分類される。

### 教師あり学習
様々な入力に対する出力を学習させ、未知の入力に対する出力を推定させる。住宅の価格を推定したり、ワインの出来を推定したり、手書き文字を認識するのは教師あり学習で実現する。

### 教師なし学習
高次元のデータを整理し、低次元のデータに射影したり（次元削減）、データをカテゴリ分けする場合に使う。アヤメの分類なんかは教師なし学習で実現する。

### 強化学習
報酬を最大化するために取るべきアクションを学習すること。ゲームを攻略するプログラムだったり、囲碁や将棋のAIと呼ばれるものは強化学習をベースにしている。

## scikit-learnのチートシート
もともと機械学習に特化したライブラリなので、基本的なところは押さえられている。詳細な説明については「[Scikit-learnとは？5分で分かるScikit-learnでできることまとめ](https://ai-kenkyujo.com/2019/07/08/can-do-with-scikit-learn/)」に譲るとして、まずは基本的なアルゴリズムから考える。

![チートシート](https://scikit-learn.org/stable/_static/ml_map.png)

### 回帰(regression)
回帰分析もたくさんあるんだが、

* 線形回帰
    * [単回帰](https://qiita.com/hiro88hyo/items/6f06b72dbbea10deb807)
    * [重回帰](https://qiita.com/hiro88hyo/items/5de0d60e9fb1c970d157)
* [基底関数回帰](https://qiita.com/hiro88hyo/items/11ef220db50a87545027)
* [勾配降下法](https://qiita.com/hiro88hyo/items/b9f449f6d0139849e7a2)
* [正則化：Lasso回帰/Ridge回帰/ElasticNet回帰](https://qiita.com/hiro88hyo/items/d467fa55d7141e8c06e5)


あたり。

### 分類(classification)
犬か猫か鳥か判別したり、文字認識をする

* 2クラス分類
    * [単純パーセプトロン](https://qiita.com/hiro88hyo/items/a81e8e1211e0cd611cf8)
    * [ロジスティック回帰](https://qiita.com/hiro88hyo/items/f5ea62a7e065bef83b6a)
    * サポートベクターマシン
        * [基本編](https://qiita.com/hiro88hyo/items/d17cb02b7356f07d16fb) 
        * [応用編](https://qiita.com/hiro88hyo/items/45772ea5636bda7faf02)
* 多クラス分類
    * 2クラス分類の多クラス化
      * [理論編](https://qiita.com/hiro88hyo/items/683d7d9feab0f33d69ea)
      * [実装編](https://qiita.com/hiro88hyo/items/7f7904ea02dc44ba191e)
    * k近傍法(kNN法)
    * 決定木
      * バギング
      * ブースティング
* ランダムフォレスト

など。

### クラスタリング(clustering)

* k-means法

とか。

### 次元削減

* PCA (主成分分析)
* カーネル主成分分析
* Matrix Factorication

## Microsoftのチートシート
Microsoftは機械学習にとても力を入れてるんですよ、論文もたくさん出してる。

### テキスト解析(自然言語処理)
形態素解析や統計解析ベクトル化など

### 画像分類

### ランク学習

# まとめ
最終的にはニューラルネットワークやらディープラーニングについても勉強していき、kaggleなんかにも進んでいくつもりだが、まずは古典的なところからしっかりやっていきたいと思う。