# Welcome!

This is a book for the Data Science for Bio/medical class. 

## Git, github

For this class, we'll be using the version control system git and git hosting service github. 
For git, you work in a *repository*, which is basically a project directory on your
computer with some extra files that help git work. Git is then used for *version control*
so that you keep track of states of your project. Github, is a hosting service for
git repositories. Typically, you have your repository on your computer and you coordinate
it with the one on the server. Github is just one of several hosting services, bitbucket
is another, or you could even relatively easily start your own. However, github has front
end web services that allows you to interact with your remote repository easily. This is
very convenient. 

I'm not going to recreate git / github tutorials here; [here's one I recommend](https://seankross.com/the-unix-workbench/git-and-github.html) by Sean Kross. Instead, I'm going to go through a typical git / github workflow.

1. **Initialization** I almost always initialize my git repository on github with a `readme.md` file.
2. **Clone** I typically *clone* the repository to my local computer using the command line or a local git gui that works with github, like [this one](https://desktop.github.com/). Note that you only have to clone the repo once. After it's cloned
you have a full local copy of the repository.
3. **add** new files to track and **stage** them after I've worked with them.
4. **commit** the changes to the local repository with a meaningful commit message.
5. **push** the changes to the repository.
6. If there's changes on the remote repository not represented in my local repository, I **pull** those changes to my local repo.

For larger projects, you're likely  working with multiple people, some of whom you've given access to your remote repository and
some of whom you have not. The ones who can't directly push to the remote repo might have their own version of the
code and their own version on github. If they think you should incorporate those changes, they might issue a **pull request** to
you. You can then opt to pull their changes into your repo on github, then pull them from github to your local repo. One of the
reasons why services like github and bitbucket are so popular is that they make this coordination fairly easy, along with having
nice project messaging and management tools. 

In our class, we use github classroom. For github classroom, you'll get a link to a repo to put your submission files into. When you push to the remote repository, you'll have submitted. *But, up to the due date you can make changes an push again.* 

