---
title: 远程连接X11VNC+SSH
date: 2024-10-17 22:27:54
tags:
    - Archlinux
    - 学习笔记
---
对于实现Linux的远程连接，使用X11VNC是一个极好的选择。曾使用过tigerVNC和X0VNC，tigerVNC本身使用必须退出当前桌面，或者为此额外开创个一个用户以使用桌面，会达到文件隔离的效果，并不适合个人用户的长期使用，而对于X0VNC他本身在权限使用方面不够方面。
作为个人用户，使用最好能类似于向日葵、ToDesk等软件，直接对物理桌面进行操控，虽然对于使用虚拟桌面有诸多缺点，但是是实际能满足个人需求的方式。
首先对于Archlinux，进行X11VNC安装。
```shell
yay -S x11vnc
```
在WIKI中，提供了Xinetd和systemd两种方法来管理X11VNC，经过个人实验，使用Xinetd管理X11VNC进程可能会在Windows上显示`not RFB server`，不清楚其中具体问题，所以直接将X11VNC嵌入为一个systemd来进行管理和开机自启。

首先需要确定当前桌面的授权文件是什么，这是关键中的关键。当其作为一个服务时，其必须确定其授权文件的具体位置，注意，不要使用`-auth guess`。在WIKI中指出授权文件地址一般为`/home/user/.Xauthority`，但是实际上在我使用的KDE+SDDM桌面环境中，授权文件并非如此，其位置在
`/run/sddm/xauth_twJiKL`中，具体如何推荐各位具体去寻找。
继续参考WIKI给出的使用方法，稍微进行修改进行实践。
先使用`vncpasswd`工具生成密码文件
```shell
vncpasswd  pwd_path(替换为你指定的存放密码的位置)
```
然后编写服务文件
```shell
sudo nano cat /etc/xinetd.d/x11vnc
```
将文件内容填写为
```shell
service x11vncservice
{
       port            = 5900
       type            = UNLISTED
       socket_type     = stream
       protocol        = tcp
       wait            = no
       user            = root
       server          = /usr/bin/x11vnc
       server_args     = -inetd e -display :0 -auth /run/sddm/xauth_twJiKL -rfbauth pwd_pathpassward_path -rfbport 5901
       disable         = no
}
```
然后重新载入，启动服务并展现服务状态
```shell
sudo systemctl daemon-reload
sudo systemctl start x11vnc.service
sudo systemctl status x11vnc.service
```
如果服务正常启动，就可以先尝试连接啦。本身在Windows客户端推荐使用 VNCviewer进行使用，可以很好的调整桌面的大小，同时连接方便。
首先查看服务器端电脑的IP
```shell
ip a
```
很多时候电脑可能只会被分配一个局域网地址，这说明你目前只可以在同一局域网中连接上该电脑。在刚刚的文件编写中已经确定服务器端电脑的服务端口在 `5901`,所以在连接框中键入`ip::port`,就可以实现连接啦。

但在实际使用中，一直开启X11VNC服务会可能产生轻微的资源占用同时不够安全，而且本质目的是远程控制个人计算机，所以使用SSH也是重要的一个工具，所以推荐使用SSH来首先进行服务器端主机连接，然后启动服务来再来使用`X11VNC`远程控制桌面，当然也可以使用
```shell
sudo systemctl enable x11vnc.service
```
来开机自启动X11VNC。

服务端使用SSH相对而言非常简单，相对而言就是直接安装，纵享丝滑。
```shell
sudo pacman -S openssh 
```
查看ssh的运行状态
```shell
systemctl status sshd
```
启动ssh
```shell
systemctl start sshd
```
设置开机自启动
```shell
systemctl enable sshd
```
如果需要停止ssh，则使用
```shell
systemctl stop sshd
```
为了连接电脑ssh，首先需要获取服务器端IP地址，上文中已经使用`ip a`获取过。
于是乎，在ssh客户端电脑上使用ssh服务(个人会使用万能编辑器VSCODE)，键入 `user@IP`
其中，作为个人PC使用，所以我的`user`选择了超级用户(本身也就只创了一个用户)。键入后输入密码，即电脑该用户密码，就可以实现访问了。

在一般使用中，通过ssh连接上电脑，然后使用 
```shell
systemctl start x11vnc
```
开启主机端的服务之后再通过VNCviewer连接进行查看，至此远程连接就做好了。
