hash = function(string, mod) sum(utf8ToInt(string)) %% mod
hash("Bruce", 5)
hash("Bruce2", 5)
hash("Haley", 5)

hash_table = vector(mode = "list", length = 5)

add_pair = function(key, value, hash_table){
    n = length(hash_table)
    new_entry = list(c(key, value))
    hash_value = hash(key, n)
    hash_entry = hash_table[[hash_value]]
    if (is.null(hash_entry)){
        hash_table[[hash_value]] = new_entry
    }
    else {
        hash_table[[hash_value]] = c(hash_entry, new_entry)
    }
    return(hash_table)
}
hash_table = add_pair("Bruce", "From individuals to populations", hash_table)
hash_table = add_pair("Bruce2", "From individuals to populations2", hash_table)
hash_table = add_pair("Haley", "Statistical methods", hash_table)
hash_table

retrieve = function(key, hash_table){
    n = length(hash_table)
    hash_value = hash(key, n)
    hash_entry = hash_table[[hash_value]]
    ## If there's nothing there return null
    if (is.null(hash_entry)){
        return(NULL)
    }
    else {
        keys = sapply(hash_entry, function(x) x[1])
        key_test = key == keys
        if (any(key_test)){
            key_index = which(key_test) 
            return(hash_entry[[key_index]][2])
        }
        ## If the key isn't there return null
        else return(NULL)
    }
}
retrieve("Bruce", hash_table)
retrieve("Bruce2", hash_table)
retrieve("bruce", hash_table)

