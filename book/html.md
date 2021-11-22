# HTML, CSS and javascript

## HTML

HTML is a markup language used by web browsers. HTML stands for *hypetext markup language*. Like all markup languages, it gives a text set of instructions that get interpreted into a nicer looking document. Other markup languages include XML, LaTeX, Org and markdown. (Yes, mark"down" is named as such since it's a ultra-simple mark"up" language.)

We'll need a little html knowledge since so much data science output is web-page oriented. Also, we'll need to know a little about html to scrape web content. A web page typically has three elements: the html which gives the page structure and markup, css (cascading style sheets) for style and javascript for interactivity. We'll cover a little of these so that we can better understand certain data science products. However, you should take a web development course if you want in depth treatments.

An HTML document looks something like this. Take a file, insert the following code and give it the extension `.html`. Then, open it up in a browser.

```
<!DOCTYPE>
<HTML>
    <HEAD>
        <TITLE> This is the web page title</TITLE>
    </HEAD>
    <BODY>
        <H1>Heading 1</H1>
        <H2>Heading 2</H2>
        <P> Paragraph </P>
        <CODE> CODE </CODE>
    </BODY>
</HTML>
```

The resulting document will look like the following

```{note}
<H1>Heading 1</H1>
<H2>Heading 2</H2>
<P> Paragraph </P>
<CODE>CODE </CODE>
```
As you probably noticed, a bit of markup is something like `<COMMAND>CONTENT</COMMAND>`. The latter command has a forward slash. 

## Browser stuff
Note, since we'll be working a lot with files, probably in one directory, you can use `file:///PATH TO YOUR DIRECTORY` to open up files (maybe even bookmark that directory). Also, `CTRL-R` is probably faster than clicking refresh and (in chrome at least) `CTRL-I` brings up developer tools (javascript console). When we have a web server running locallly, you usually go to `localhost`. For example, my jupyter lab server sends me to `http://localhost:8888/lab/tree/`. Here `8888` is a port, `localhost` refers to the server running on the lcoal computer.

Browsers make choices in how they render HTML and implement javascript. So, unless you're a web developer by trade, don't get too exotic in your design choices.

## Hosting
When you double click on your html file, it's being hosted locally. So, no one else can see it. To have a web page on the internet it has to be hsoted on a server running web hosting software. Fortunately, github will actually allow us to host web pages. Basically, put an empty `.nojekyll` file in your repository (this tells it that it's not a jekyll based web site and follow the instructions [here](https://pages.github.com/). This will be really useful for us, since many of our datascience programs output web pages. For example, RMarkdown documents get translated into web documents. Similarly, jupyter-lab will output reveal.js (javascript/html) slide decks from our jupyter lab notebooks. Note that some of our programs will require servers that also run python or R in the back end, so github pages won't suffice for that. There we need servers specifically set up to run those kinds of scripts.

## Javascript

Javascript is what makes webpages interactive. We'll need a little javascript to develop interactive graphics. Consider the following where we use javscript to change an HTML element in a web page

```
<H2 id="textToChange">Preference ?</H2>

<button type="button" onclick='document.getElementById("textToChange").innerHTML = "You prefer 1"'>1</button>
<button type="button" onclick='document.getElementById("textToChange").innerHTML = "You prefer 2"'>2</button>
```

```{note}
<H2 id="textToChange">Preference ?</H2>
<button type="button" onclick='document.getElementById("textToChange").innerHTML = "You prefer 1"'>Click 1</button>
<button type="button" onclick='document.getElementById("textToChange").innerHTML = "You prefer 2"'>Click 2</button>
```




