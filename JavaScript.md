JavaScript
===
* 保存书签
    `CTRL+shift+1`
* 跳转到标签
    `Ctrl+1`
---

* 删除node_modules 快捷方法
  `npm i rimraf -g`
  `rimraf node_modules`

* 解决github上拉下来的项目运行报错的问题
  出现原因：npm的第三方依赖出现了 冲突
  解决：
  * 在可以运行的项目内运行`npm shrinkwrap`生成一个新的依赖锁文件
  * 将新生成的`npm-shrinkwrap.json`文件和`package.json`文件一起复制到新的项目中，替换掉原来的`package.json`和`package-lock.json`
  * 运行`npm ci`安装依赖

* 报错：`This is probably not a problem with npm. There is likely additional logging output above.`
    删除node_modules重新安装即可

  （注意：`npm install`命令会rewrite `package-lock.json`文件，一定要小心）

* 使用`guans-style`
  * import guans-style
  * Tutorial：https://gitee.com/shaolunryan/GUANS-libs/tree/master/guans-style
  * bootstrap中文文档：https://getbootstrap.net/docs/getting-started/introduction/



* datGUI的使用
  * `npm i react-dat-gui -S`
  
  ```js

  import React from 'react';
  import DatGui, { DatBoolean, DatColor, DatNumber, DatString } from 'react-dat-gui';

  class App extends React.Component {
    state = {
      data: {
        package: 'react-dat-gui',
        power: 9000,
        isAwesome: true,
        feelsLike: '#2FA1D6',
      }
    }

  // Update current state with changes from controls
  handleUpdate = newData =>
    this.setState(prevState => ({
      data: { ...prevState.data, ...newData }
    }));

  render() {
    const { data } = this.state;

    return (
      <DatGui data={data} onUpdate={this.handleUpdate}>
        <DatString path='package' label='Package' />
        <DatNumber path='power' label='Power' min={9000} max={9999} step={1} />
        <DatBoolean path='isAwesome' label='Awesome?' />
        <DatColor path='feelsLike' label='Feels Like' />
      </DatGui>
    )
  }
  ```
  文档：https://github.com/claus/react-dat-gui


* react的生命周期
  省去不常用的part，最常用的顺序如下所示：
  `state挂载->DOM的挂载->componentDidMount()函数`

  ---
部署项目时打包静态文件路径的问题
* 用`create-react-app`创建的项目，要请求的静态数据要放在`public`文件夹中

    经测试，使用create-react-app创建的项目，在引入静态数据文件时，显示正常
    引入方式：`axios.get('data/data.json')`
    data位置：`/public/data/data.json`
    package.json中的`homepage`配置：None


* 用`webpack`打包的项目
    成功取到数据文件
    方式：其实很简单，就是要明确一个理解：
    项目ip：131.123.39.100的`root`，就是 项目的dist文件夹
    这个在nginx的配置文件中也指定过
    所以在代码中就这样写：
    ```js
      axios.get('statics/data.json')
      .then()
      .then()
    ```
    然后把`data.json`放在`dist/statics/`下即可