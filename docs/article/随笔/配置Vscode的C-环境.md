---
title: 配置Vscode的C++环境(转载)
comments: true
copyright: false
date: 2020-02-18 15:16:48
tags: 配置环境
categories:
	- 随笔
photo: http://ww1.sinaimg.cn/large/006eb5E0gy1gc0l1ao9toj308m04jq52.jpg
top:
cover: https://api.ixiaowai.cn/mcapi/mcapi.php
keyword:  
toc: true
---



# [Visual Studio Code配置C/C++编译运行环境](https://www.cyprestar.com/2019/03/11/Configure-vscode-for-cpp/)

最近一直在用C++写LeetCode，本地调试的时候一直在纠结用什么编辑器，CLion有点耗资源，Visual Studio又有点杀鸡焉用宰牛刀，Dev-Cpp已经太古老了，于是决定用VS Code来写，但是官方文档写的有点……不是特别友好，因此在这里记录一下自己是怎么配置的。

<!--more-->

- [1 安装MinGW或者CLang](https://www.cyprestar.com/2019/03/11/Configure-vscode-for-cpp/#1-安装MinGW或者Clang)
- 2 安装并配置C/C++插件
  - [2.1 配置IntelliSence](https://www.cyprestar.com/2019/03/11/Configure-vscode-for-cpp/#2-1-配置IntelliSence)
  - [2.2 配置编译选项](https://www.cyprestar.com/2019/03/11/Configure-vscode-for-cpp/#2-2-配置编译选项)
  - [2.3 配置Debug选项](https://www.cyprestar.com/2019/03/11/Configure-vscode-for-cpp/#2-3-配置Debug选项)
  - [2.4 设置代码格式化](https://www.cyprestar.com/2019/03/11/Configure-vscode-for-cpp/#2-4-设置代码格式化)
- [3 后续](https://www.cyprestar.com/2019/03/11/Configure-vscode-for-cpp/#3-后续)

### 1 安装MinGW或者Clang

目前Windows上比较主流的C++编译器包括[MinGW](https://sourceforge.net/projects/mingw-w64/files/)（也就是Windows版本的GCC）以及[CLang](http://releases.llvm.org/download.html)。在这里以MinGW为例。

MinGW可以下载Online Installer然后选择对应的版本号进行安装，在这里可以选择`8.1.0`, `posix`和`seh`。对于不同版本的对比可以参考这篇[StackOverflow](https://stackoverflow.com/questions/29947302/meaning-of-options-in-mingw-w64-installer)。

安装完成后将`MinGW安装目录/bin`文件夹（例如`C:\Dev\mingw64\bin`）加入环境变量，具体可以参考[这篇文档](https://www.eclipse.org/4diac/documentation/html/installation/minGW.html)。加入环境变量后重新开启一个命令行然后输入`g++ --version`就可以看到安装好的MinGW的版本号了。

### 2 安装并配置C/C++插件

访问[C/C++ - Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)点击`Install`启动VS Code安装插件即可。

可以参考[官方文档](https://code.visualstudio.com/docs/languages/cpp)来配置C/C++插件，这里着重对官方文档中没有讲明白的部分进行强调。比较懒的同学可以直接跳过每一节后面详细配置部分，直接复制配置修改对应编译器和Debugger的位置即可使用。下面均以Windows系统为主，macOS和Linux可以根据详细配置部分自行修改。

#### 2.1 配置IntelliSence

新建一个项目文件夹，在VS Code中打开，然后在VS Code中按`Ctrl + Shift + P`输入命令`C/C++: Edit Configurations...`，VS Code会自动在项目文件夹根目录新建配置文件夹`.vscode`并同时在`.vscode`中新建配置文件`c_cpp_properties.json`。

先给出配置文件内容：

```json
{
    "configurations": [
        {
            "name": "Win32",
            "includePath": [
                "${workspaceFolder}/**"
            ],
            "defines": [
                "_DEBUG",
                "UNICODE",
                "_UNICODE"
            ],
            "compilerPath": "C:/Dev/MinGW/mingw64/bin/gcc.exe", //修改为本机gcc.exe路径
            "cStandard": "c11",
            "cppStandard": "c++17",
            "intelliSenseMode": "gcc-x64"
        }
    ],
    "version": 4
}
```

然后介绍其中重要的部分：

`c_cpp_properties.json`中共包含三部分：

`env`：可以在这里定义用户变量，在下面的`configuration`中可以使用这里定义的变量来代替多次重复出现的地址等配置。

`configurations`：在这里配置IntelliSence选项。每一个选项都会有默认值。

`version`：指`c_cpp_properties.json`的版本，这里不需要作修改。

`configurations`中包含以下配置项：

`name`：配置文件名称，在这里如果根据系统填”Linux”, “Mac”或者”Win32”，那么插件会自动根据这里填写的系统名称来读取默认值。在VS Code的状态栏中会显示这一配置名称。通过点击状态栏中的名称可以更换配置。在这里我们指定为`Win32`。

`intelliSenseMode`：如果插件设置中`C_Cpp.intelliSenseEngine`设置为”default”的话，这里可以指定IntelliSence的模式。`msvc-x64`对应Visual Studio模式，`clang-x64`对应Clang模式，`gcc-x64`对应GCC模式。Windows默认使用`msvc-x64`，macOS默认使用`clang-x64`，Linux默认使用`gcc-x64`，在这里由于我们用的是GCC，因此修改为`gcc-x64`。

`includePath`：指定IntelliSence在目录中搜索源文件所包含（`#include`）的头文件。这与在命令行中调用编译器时用`-I`选项指定的路径是一样的。如果路径末尾是`/**`的话IntelliSence会自动加载所有的子目录。如果是Windows系统并且安装了Visual Studio 的C++功能，或者配置了`compilePath`，这里可以不做配置。

`macFramePath`：同上，指定macOS框架的头文件目录地址。

`defines`：这里可以指定在编译过程中所调用的预处理符号的内容，与命令行调用编译器时`-D`选项指定的内容是一样的。如果是Windows系统并且安装了Visual Studio 的C++功能，或者配置了`compilePath`，这里可以不做配置。

`forceInclude`：（可选）强制加载的头文件，优先于源文件中指定的头文件，按照配置文件里的顺序加载，这里不需要配置。

`compilerPath` ：（可选）编译器的绝对路径。插件会根据编译器决定`includePath`和default `define`的值。可以在路径后附加选项，例如`-nostdinc++`, `-m32`, `-fno-ms-extensions`等等。如果选项有空格需要用双引号（`“`）括起来。这里需要找到刚才安装的MinGW中`gcc.exe`的位置，例如`C:/Dev/MinGW/mingw64/bin/gcc.exe`。

`cStandard`：C的标准，这里填`c11`。

`cppStadard`：C++的标准，这里填`cpp17`。

`compileCommands`：（可选）

> If `"C_Cpp.intelliSenseEngine"` is set to “Default” in your settings file, the includes and defines discovered in this file will be used instead of the values set for `includePath` and `defines`. If the compile commands database does not contain an entry for the translation unit that corresponds to the file you opened in the editor, then a warning message will appear and the extension will use the `includePath` and `defines` settings instead.
>
> > For more information about the file format, see the [Clang documentation](https://clang.llvm.org/docs/JSONCompilationDatabase.html). Some build systems, such as CMake, [simplify generating this file](https://cmake.org/cmake/help/v3.5/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html).

参考：

[`c_cpp_properties.json` Reference Guide](https://github.com/Microsoft/vscode-cpptools/blob/master/Documentation/LanguageServer/c_cpp_properties.json.md)

### 2.2 配置编译选项

VS Code通过[Tasks](https://code.visualstudio.com/docs/editor/tasks)来执行编译和运行操作。在`.vscode`文件夹中新建文件`tasks.json`进行配置。

先给出配置文件内容：

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build",
            "command": "C:/Dev/MinGW/mingw64/bin/g++.exe", // 修改这里为本机g++路径
            "type": "shell",
            "args": [
                "-g",
                "-Wall",
                "-std=c++11",
                "-lm",
                "${file}",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}.exe" //Linux和macOS这里要修改为.o
            ],
            "presentation": {
                "reveal": "always",
                "echo": true,
                "focus": true
            },
            "problemMatcher": {
                "owner": "cpp",
                "fileLocation": ["relative", "${workspaceRoot}"],
                "pattern": {
                    "regexp": "^(.*):(\\d+):(\\d+):\\s+(warning|error):\\s+(.*)$",
                    "file": 1,
                    "line": 2,
                    "column": 3,
                    "severity": 4,
                    "message": 5
                }
            },
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }, 
        {
            "label": "Run",
            "type": "shell",
            "dependsOn": "Build",
            "command": "${fileDirname}/${fileBasenameNoExtension}.o",
            "windows": {
                "command": "${fileDirname}/${fileBasenameNoExtension}.exe"
            },
            "args": [],
            "presentation": {
                "reveal": "always",
                "focus": true
            },
            "problemMatcher": [],
            "group": {
                "kind": "test",
                "isDefault": true
            }
        }
    ]
}
```

然后解释其中配置的内容：

`label`：任务标签。

`type`：任务类型，一般指定为`shell`或`process`。当设置为`shell`时会将命令看作终端操作，包括bash、cmd或PowerShell。当指定为`process`时会将命令看作是一个进程。这里选择`shell`。

`command`：命令内容，这里指定编译器的路径，比如g++的路径`C:/Dev/MinGW/mingw64/bin/g++.exe`。

`args`：命令的参数列表。

`windows`：Windows特定的配置属性，例如macOS和Linux上指定编译的输出文件为`.o`，而Windows下指定编译的输出文件为`.exe`。

`group`：指定任务的分组归属。例如指定`group`为`test`，那么可以通过命令面板的`Run Test Task`运行该任务。

`presentation`：指定如何在界面上显示任务输出。

 `reveal`：在执行任务时是否显示内置的终端面板，可选的值包括：`always`：始终显示；`never`：从不显示；`silent`：只在不扫描错误和警告时显示。默认值为`always`。

 `focus`：终端是否在输入时激活。默认值为`false`。

 `echo`：是否将执行的命令输出到终端中。默认值为`true`。

 `showReuseMessage`：是否显示”终端将被任务重用，按任意键关闭。“这一消息。

 `panel`：控制是否在各任务间共享终端输出。

 `shared`：共享终端，所有任务的输出都会在同一个终端内显示。

 `dedicated`：每一个任务都有自己的输出终端，但是对于同一个任务来说，如果被重复执行的话仍会在同一个终端内显示。

 `new`：无论是否是同一个任务都会打开一个新的输出终端。

 `clear`：在任务执行之前是否清空输出终端，默认值为`false`。

 `group`：指定任务输出的分组，同一个分组的任务会共享终端而非开启新终端。

`problemMatcher`：将编译器输出的错误映射至VS Code的问题面板。具体可以参考[Defining a problem matcher](https://code.visualstudio.com/docs/editor/tasks#_defining-a-problem-matcher)。

参考：

[Integrate with External Tools via Tasks](https://code.visualstudio.com/docs/editor/tasks)

[Schema for tasks.json](https://code.visualstudio.com/docs/editor/tasks-appendix)

#### 2.3 配置Debug选项

如果要启用Debug，需要在.vscode文件夹下创建`launch.json`。

先给出配置内容：

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    // https://github.com/Microsoft/vscode-cpptools/blob/master/launch.md
    "version": "0.2.0",
    "configurations": [
        {
            "name": "(gdb) Launch", // 配置名称，将会在启动配置的下拉菜单中显示
            "type": "cppdbg",// 配置类型，这里只能为cppdbg
            "request": "launch", // 请求配置类型，可以为launch（启动）或attach（附加）
            "program": "${fileDirname}/${fileBasenameNoExtension}.o", // 将要进行调试的程序的路径
            "args": [],
            "stopAtEntry": false,
            "cwd": "${workspaceFolder}",
            "externalConsole": true,
            "environment": [],
            "MIMode": "gdb",
            "windows": {
                "program": "${fileDirname}/${fileBasenameNoExtension}.exe",
                "miDebuggerPath": "C:/Dev/MinGW/mingw64/bin/gdb.exe" // 修改为本机gdb.exe的路径
            },
            "setupCommands": [
                {
                    "description": "Enable pretty-printing for gdb",
                    "text": "-enable-pretty-printing",
                    "ignoreFailures": true
                }
            ],
            "preLaunchTask": "Build"
        }
    ]

}
```

然后解释每一项具体的含义：

`program`：（必填项）指定Debugger程序路径，在这里修改成本机gdb.exe的路径。

`symbolSearchPath`：指定符号文件的路径。

`externalConsole`：在Windows中，设置为`true`调用外部终端，设置为`false`调用VS Code集成终端；在Linux中，设置为`true`调用会通知VS Code启动外部终端，设置为`false`会调用VS Code集成终端；在macOS中，设置为`true`会通过`lldb-mi`调用外部终端，设置为`false`会在Debug面板中显示输出。

`args`：参数列表。

`cwd`：终端启动时所在的路径。

`environment`：环境变量。

`MIMode`：指定VS Code连接的Debugger的类型，必须是`gdb`或者`lldb`中一种。

`miDebuggerPath`：Debugger路径。如果没有指定完整路径，VS Code会搜索系统变量`PATH`。

`miDebuggerArgs`：传递给Debugger的参数列表。

`stopAtEntry`：是否在程序入口暂停。默认值为`false`。

`type`：指定Debugger类型，对于Visual Studio Windows debugger需要指定为`cppvsdbg`，对于GDB或者LLDB需要指定为`cppdbg`。

参考：

[Configuring `launch.json` for C/C++ debugging](https://github.com/Microsoft/vscode-cpptools/blob/master/launch.md)

#### 2.4 设置代码格式化

在VS Code里可以通过`Shift + ALt + F`对整篇代码进行格式化，或者选中一段代码通过`Ctrl + K Ctrl + F`来进行段落的格式化。也可以在设置中设置`editor.formatOnSave`设定在保存时自动格式化或`editor.formatOnType`设定在输入时自动格式化。默认VS Code调用Clang Format对代码进行格式化，因此在项目文件夹下新建`.clang-format`文件来指定格式化样式。由于Clang Format可以设置的项目非常多，因此这里只举例说明：

```c++
UseTab: false
IndentWidth: 4
BreakBeforeBraces: Stroustrup 
AllowShortIfStatementsOnASingleLine: false
IndentCaseLabels: true
ColumnLimit: 0
Standard: Cpp11
Cpp11BracedListStyle: true
```

`UseTab`：用Tab或空格控制缩进。

`IndentWidth`：缩进长度。

`BreakBeforeBraces`：在括号前是否换行。

`AllowShortIfStatementsOnASingleLine`：是否将 `if (a) return;`放在同一行。

`IndentCaseLabels`：对于`switch`语句，`case`是否缩进。

`ColumnLimit`：限制一行的长度。

`Standard`：cpp标准，`Cpp03`会适应C++03标准，`cpp11`会适应C++11、C++14和C++17标准。例如， C++03标准的`A >`会被替换为C++11标准的`A>`。

`Cpp11BracedListStyle`：使用C++11的列表格式：

```c++
true:                                  false:
vector<int> x{1, 2, 3, 4};     vs.     vector<int> x{ 1, 2, 3, 4 };
vector<T> x{{}, {}, {}, {}};           vector<T> x{ {}, {}, {}, {} };
f(MyMap[{composite, key}]);            f(MyMap[{ composite, key }]);
new int[3]{1, 2, 3};                   new int[3]{ 1, 2, 3 };
```

参考：[Clang-Format Style Options](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)

### 3 后续

完成以上步骤以后你会得到一个`.vscode`文件夹（包含三个配置文件）以及`.clang-format文件`。之后再新建项目时直接将这个文件夹复制过去即可。

打开一个`.cpp`文件然后按`F5`即可启动调试。按`Ctrl + Shift + P`然后输入`Run Build Task`即可编译。输入`Run Test Task`即可运行。在设置中可以为这两个选项添加快捷键绑定。按`Ctrl + Shift + F`（我这里 shift + Alt + F）可以格式化代码。

以上。

### 参考

[VS Code 搭建 C/C++ 编译运行环境的三种方案](https://zhuanlan.zhihu.com/p/35178331)

[C/C++ for Visual Studio Code (Preview)](https://code.visualstudio.com/docs/languages/cpp)

[`c_cpp_properties.json` Reference Guide](https://github.com/Microsoft/vscode-cpptools/blob/master/Documentation/LanguageServer/c_cpp_properties.json.md)

[Configuring `launch.json` for C/C++ debugging](https://github.com/Microsoft/vscode-cpptools/blob/master/launch.md)

[Variables Reference](https://code.visualstudio.com/docs/editor/variables-reference)

[Customizing IntelliSense](https://code.visualstudio.com/docs/editor/intellisense#_customizing-intellisense)

[Integrate with External Tools via Tasks](https://code.visualstudio.com/docs/editor/tasks)

[Clang-Format Style Options](https://clang.llvm.org/docs/ClangFormatStyleOptions.html)























