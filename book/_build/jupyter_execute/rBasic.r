# Comments in R are the # sign, just like python
# Arithmetic works like you would expect
1 + 2 * 3

# Assign a variable, note y is a separate entity than x (run this same example in python)
x = 5
y = x
x = 10
y

# list out our variables that we've created
    ls()

z = c(1, 5, 8)
# Most operations are elementwise
z + 5 * z

z + c(z, z)

3 == 4
3 == 3

for (i in 1 : 6){
    if (i <= 3) {
        print("i is small")
    }
    else {
        print("i is large")
    }        
}



x = list(a = 1 : 3, b = "character", c = list(a = 1 : 4, b = "character2"))

x$a
x$b
x[[1]]
x[[2]]

x = data.frame(index = 3 : 7, letter = letters[3 : 7])
x

x[,1]
x[1 : 2,]
x[1,2]

x = matrix( 1 : 6, 3, 2)
x
y = matrix( 1 : 6, 2, 3)
y

x[1,]
x[,1]
x[1, 2]

pow = function(x, n) {
    x ^ n
}
pow(2, 3)
pow(x = 2, n = 3)
pow(n = 3, x = 2)
pow(n = 3, 2)
pow(3, 2)

doublefunc = function(f, x, ...){
    f(x, ...) * 2
}
doublefunc(pow, 2, 3)
doublefunc(exp, 2)

a = 2
f = function(b){
    c1 = 3
    g = function(d){
        e = 4
        return(1)
    }
    print(e)
    return(1)
}
print(c1)

f(1)
