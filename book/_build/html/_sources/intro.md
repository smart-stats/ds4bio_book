# Welcome!

This is a book for the Data Science for Bio/Public Health/medical classes. 

## Markdown
Before getting started, you need to learn some markdown. Markdown is a markup language (like HTML) that is absurdly easy. Every data scientist needs to know markdown. Fortunately, you're five minutes away from knowing it. A markdown file is a text file that needs to be rendered to look nice. If you want an example, this page was written in markdown. To try it, go to google colab, create a markdown cell and start editing. [Try this cheat sheet](https://www.markdownguide.org/cheat-sheet/).

## Some basic unix

Some basic unix commands will go a long way in the course, especially for when you're working on a remote server. On windows, you can actually install a unix environment on so-called services for unix.  So, on windows, you have three options for working with the command line, i) install a linux subsystem and use that, ii) use the DOS command prompt or iii) use powershell. The commands here would only work for option i. However, when I work on a windows system, I tend to just use options ii or iii. Here, we'll assume you're working on a linux or unix system, or windows services for linux and you'll have to read up elsewhere if you want to learn windows proper terminal commands. 

To get a unix terminal, you have several options. Since we're promoting jupyter and jupyterlab, just open up the terminal on there. (Again, assuming you're working on a unix/linux system.)

The first thing you should try is figuring out where you're at. Do this by typing
```
prompt> pwd
```
This will show you where you are in the directory structure. If you want to see the contents of the directory try these 
```
prompt> ls
prompt> ls -al
prompt> ls -alh
```
Adding the flags `-a` lists everything, including directories with a weird character in front. The `l` gives the long format, which gives more information and the `h` changes the filesize lists to a more human readable format.  I also like the option `--color`. `What you get with `-alh` is as follows.

```
total 36K
drwxrwxrwx+ 7 codespace root      4.0K Feb 14 14:24 .
drwxr-xrwx+ 5 codespace root      4.0K Oct 19 15:21 ..
drwxrwxrwx+ 6 codespace codespace 4.0K Feb 14 14:31 book
drwxrwxrwx+ 8 codespace root      4.0K Feb 15 21:34 .git
-rw-rw-rw-  1 codespace codespace  171 Feb 14 14:24 .gitignore
-rw-rw-rw-  1 codespace codespace    0 Feb 14 14:23 .nojekyll
-rw-rw-rw-  1 codespace codespace  444 Feb 14 14:24 README.md
drwxrwxrwx+ 3 codespace codespace 4.0K Feb 14 14:24 slides
drwxrwxrwx+ 7 codespace codespace 4.0K Oct 19 15:21 .venv
drwxrwxrwx+ 2 codespace codespace 4.0K Oct 19 15:23 .vscode
```
The  `drwxrwxrwx+1` looking columns give permissions d=directory, r=read, w=write and x=execute, the groups are owner (you), group, everyone. So a file that is
`-rw-------` can be read and written to by the owner, but cannot be executed by anyone and no one else can read or write to it (except the superuser, who gets to do everything). 

To change a directory, try the following
```
prompt> cd DIRECTORY
```
where `DIRECTORY` is the name of the directory that you want to change into. You can hit TAB to autocomplete names. The command
```
prompt> mv PATH_TO_INPUT_FILE PATH_TO_OUTPUT_FILE
```
moves the file. This is also how you rename a file, since you could just do `mv FILENAME1 FILENAME2` and change the name.

The unix command for removing things is `rm`. So
```
rm FILENAME
```
deletes the file. Note linux really deletes things, so do this with some care. You can't remove directories this way, instead you could do `rmdir DIRECTORY`, but the directory has to be empty. If you want to use `rm` to remove a directory and its contents, you can do `rm -rf DIRECTORY`. However, use this with care.

Finally, I find it very useful to use `wget` to grab files from the internet. So, for example,
```
wget https://URL.../FILENAME
```
will grab the file from that link. Super useful.

That's enough unix to get you started. You'll find as you use the terminal more and more, you'll like it better and better. Eventually, you'll find GUIs kind of frustrating. 

