git & github
===
第一次绑定一个项目到github
---
* `git init`
* `git add .`
* `git commit -m 'my_message'`
* 在github创建一个仓库
* `git remote add origin http://...`
* `git push -u origin master`

* 生成公钥命令
`ssh-keygen -t rsa -C 'haywardryan@foxmail.com'
* 位置
`/home/shaolun/.ssh`
* 文件
`id_rsa.pub`

* github无法访问
  直接输入`http://github.com`
  ~~因为已经添加到hosts文件中了，所以直接输入网址访问理论上是可以访问的~~
  直接换成手机热点连接

---

overleaf
===

* 实用的快捷键
  `Ctrl+z` 撤销
  `Ctrl+y` 反撤销
  `Ctrl+Delete` 删除一行
  `Ctrl+L` 添加\跳转到书签

* 定义工具名称
  `\newcommand{\toolName}{\textit{Delta Map}}`
* 定义批注的名字
  `\usepackage{xcolor}`
  `\newcommand{\shaolun}[1]{\textcolor{blue}{\textit{Shaolun:#1}}}`
* 加标签+后文引用
  -`\label{}`
    -`\ref{}`

* 加图片
  ```latex
  \begin{figure}[tbhp]
  \centering 
  \includegraphics[width=\columnwidth]{figures/Fig-all_bar_charts.pdf}
  \label{fig:Fig-all_bar_charts}
  \caption{}
  \end{figure}
  ```

* 加item
  ```latex
  \begin{itemize}
  \item hahahah
  \end{itemize}
  ```

--- 
markdown
===
* 用于删除的中划线
  `~~我是要删除的文字~~`
  效果：~~我是要删除的文字~~