# Problem sets from CCPS109

import itertools as it
import collections as co
import numpy as np
import string

def ryerson_letter_grade(pct):
    if pct >= 90:
        return "A+"
    elif 85 <= pct <= 89:
        return "A"
    elif 80 <= pct <= 84:
        return "A-"
    elif 77 <= pct <= 79:
        return "B+"
    elif 73 <= pct <= 76:
        return "B"
    elif 70 <= pct <= 72:
        return "B-"
    elif 67 <= pct <= 69:
        return "C+"
    elif 63 <= pct <= 66:
        return "C"
    elif 60 <= pct <= 62:
        return "C-"
    elif 57 <= pct <= 59:
        return "D+"
    elif 53 <= pct <= 56:
        return "D"
    elif 50 <= pct <= 52:
        return "D-"
    else:
        return "F"
    
def is_ascending(x):
    i = sorted(x)
    n = 0
    if i != x: 
        return False
    for n in range(len(x)-1): 
        if x[n] == x[n+1]:
            return False
        else:
            n += 1
    return True

def is_ascending_rec(items):
    if len(items) <= 1:
        return True
    if items[0] >= items[1]:
        return False
    return is_ascending_rec(items[1:])

def riffle(items, out = True):
    a = items[:len(items) // 2]
    b = items[len(items) // 2:]
    items = a + b
    if not out:
        items[::2] = b
        items[1::2] = a
    else:
        items[::2] = a
        items[1::2] = b
    return items

def is_cyclops(n):
    g = str(n)
    a = g.split('0')
    if g.count('0') != 1:
        return False
    elif len(a[0]) == len(a[1]):
        return True
    return False

def only_odd_digits(n):
    s = str(n)
    if s.count('0') > 0:
        return False
    elif s.count('2') > 0:
        return False
    elif s.count('4') > 0: 
        return False
    elif s.count('6') > 0: 
        return False
    elif s.count('8') > 0:
        return False
    return True

def only_odd_digits_rec(n):
    if n < 10:
        return n % 2 == 1
    if (n % 10) % 2 == 0:
        return False
    return only_odd_digits_rec(n // 10)
    
#def pyramid_blocks(n, m, h):
   # a = n*m
    #for i in range(0, h-1):
    #    n += 1
    #    m += 1
    #    a += n*m
    #return a
    
def domino_cycle(tiles):
    li = [item for t in tiles for item in t] 
    for i in range(0, len(li), 2):
        if li[i] != li[i-1]:
            return False
    return True

def two_summers(items, goal):
    i, j = 0, len(items)-1
    while i < j:
        x = items[i] + items[j]
        if x == goal:
            return True
        elif x > goal:
            j -= 1
        elif x < goal:
            i += 1
    return False

def three_summers(items, goal):
    if items[-1] + items[-2] + items[-3] < goal:
        return False
    for i in range(len(items)):
        tmp = items[:]
        del(tmp[i])
        if two_summers(tmp, goal-items[i]):
            return True
    return False
    
def nearest_smaller(items):
    res = []
    n = len(items)
    
    for i, e in enumerate(items):
        j = 1
        while j < n: 
            left = items[i-j] if i >= j else e
            right = items[i+j] if i+j < n else e
            
            if left < e or right < e:
                res.append(min(left, right))
                break
            j += 1
        else:
            res.append(e)
    return res    

def count_and_say(digits):
    if digits == '': return ''

    res = ''
    curr = digits[0]
    nc = 1
    for c in digits[1:]:
        if c != curr:
            res += str(nc) + curr
            nc = 1
            curr = c
        else:
            nc += 1
    res += str(nc) + curr
    return res
        
def reverse_vowels(text):
    result = ""
    vowels = [c for c in text if c in 'aeiouAEIOU']
    
    for ch in text:
        if ch not in 'aieouAEIOU':
            result += ch
        else:
            c = vowels.pop()
            if ch in 'AEIOU':
                result += c.upper()
            else:
                result += c.lower()
    return result

def create_zigzag(rows, cols, start = 1):
    li = []
    n = start
    for r in range(rows):
        li.append(list(range(n, n+cols)))
        n += cols
    for r in range(1,rows,2):
        li[r].reverse()
    return li

def bulgarian_solitaire(piles, k):
    count = 0
    goal = list(range(1, k+1))
    while True:
        if len(piles) == k:
            if sorted(piles) == goal:
                return count
        piles = [p-1 for p in piles if p>1]+[len(piles)]
        count += 1
    return count    

def count_dominators_shlemiel(items):
    if items == []:
        return 0
    count = 1
    for a in range(len(items) - 1):
        for b in range(a+1, len(items)):
            if items[b] >= items[a]:
                break
            else:
                count += 1
    return count

def count_dominators(items):
    total, big = 0, None
    for e in reversed(items):
        if big == None or e > big:
            big = e
            total += 1
    return total

def word_match(w, letters):
    for l in letters:
        i = w.find(l)
        if i < 0:
            return False
        w = w[i+1:]
    return True

def words_with_letters(words, letters):
    return [w for w in words if word_match(w, letters)]

def detab(text, n = 8, sub = ' '):
    return text.expandtabs(n)

def expand_intervals_slower(intervals):
    result = []
    for i in intervals.split(','):
        p = i.split('-')
        if len(p) == 2:
            result.extend(range(int(p[0]), int(p[1]) + 1))
        else:
            result.append(int(p[0]))
    return result

def expand_intervals(intervals):
    result = []
    for i in intervals.split(','):
        p = i.partition('-')
        if p[1] == '-':
            result.extend(range(int(p[0]), int(p[2]) + 1))
        else:
            result.append(int(p[0]))
    return result

#def is_hangman_word(w, pattern):
#   if len(w) == len(pattern):
#        return False
#    for (cw, cp) in zip(w, pattern):
#        if cp != '*' and cp != cw:
#            return False
#        if cp == '*' and cw in pattern:
#            return False
#    return True

#def possible_words(words, pattern):
#    return [w for w in words if is_hangman_word(w, pattern)]

def possible_words(words, pattern):
    result = []
    for w in words:
        if len(w) == len(pattern):
            for (cw, cp) in zip(w, pattern):
                if cp != '*' and cp != cw:
                    break
                if cp == '*' and cw in pattern:
                    break
            else: 
                result.append(w)
    return result

def double_until_all_digits(n, giveup = 1000):
    
    count = 0
    m = n
    while count != giveup:
        st = str(m)
        if st.count('0') >= 1 and st.count('1') >= 1 and st.count('2') >= 1 and st.count('3') >= 1 and st.count('4') >= 1 and st.count('5') >= 1 and st.count('6') >= 1 and st.count('7') >= 1 and st.count('8') >= 1 and st.count('9') >= 1:
            return count
            break
        else:
            m = m * 2
            count += 1
    return -1

def duplicate_digit_bonus(n):
    m = str(n)
    count = {} #dictionary to capture each block of digits and count k digits per block
    key = 0
    score = 0
    count[key] = 1
    for i in range(len(m)-1):
        if m[i] == m[i+1]: #check next element to see if it is a part of the block
            count[key] += 1
        else: #block split, next dict key and initialize key value
            key += 1
            count[key] = 1
    for k in count:
        if count[k] > 1:
            score = score + 10**(count[k]-2)
    if count[list(count)[-1]] > 1:
        score = score + (10**(count[list(count)[-1]]-2))
    return score
           

def pancake_scramble(text):
    sc = text
    for n in range(len(text)+1):
        scramble = sc[0:n][::-1] + sc[n:]
        sc = scramble
    return scramble

def sum_of_two_squares(n):
    small = 1 #similar to three_summers(two_summers) going outward in to find the squares (which also solves the condition for largest possible square)
    large = 1 
    while large*large < n:
        large = large + 1
    while large >= small:
        sum = small*small + large*large
        if sum == n:
            return (large, small)
        elif sum > n:
            large -= 1
        elif sum < n:
            small += 1
    return None

def hitting_integer_powers(a, b, tolerance = 100):
    pa = 1
    pb = 1
    aa = a**pa
    bb = b**pb
    
    while tolerance * abs(aa-bb) > min(aa, bb):
        if aa < bb:
            aa = aa * a #effectively increasing pa by 1 without doing exponentiation 
            pa += 1
            #pa += 1
            #aa = a**pa #if done this way, would be very inefficient
        else:
            bb = bb * b
            pb += 1
    return (pa, pb)

def reverse_ascending_sublists(items):
    asc = [] #try with dictionary
    result = []
    for e in (items + [None]):
        if e != None and (asc == [] or asc[-1] < e):
            asc.append(e)
        else:
            asc.reverse()
            result.extend(asc)
            asc = [e]
    return result
   

def taxi_zum_zum(moves):
    counter = 0
    x = 0 
    y = 0
    for m in moves:
        if m == 'L' and counter > 0:
            counter -= 1
        elif m == 'L' and counter == 0:
            counter = 3
        elif m == 'R' and counter < 3:
            counter += 1
        elif m == 'R' and counter == 3:
            counter = 0
        else:
            if counter == 0:
                y += 1
            elif counter == 1:
                x += 1
            elif counter == 2:
                y -= 1
            elif counter == 3:
                x -= 1
    return (x, y)
       

def extract_increasing(digits):
    li = digits[1:]
    result = [int(digits[0])] #the first digit from the input will always be the first in the list
    while len(str(result[-1])) <= len(li): #as long as the length of the list is greater than the last result digit, there will be enough digits
        length = len(str(result[-1])) #length of last element in result list to compare to input list
        block = li[0:length]
        if length == len(li) and int(result[-1]) >= int(li):
            break
        if (int(result[-1]) >= int(block)):
            if block[0] == '0':
                li = li[1:]
            else:
                result.append(int(li[0:length+1]))
                li = li[length+1:]
        else:
            result.append(int(block))
            li = li[length:]
    return result
    
def running_median_of_three(items):
    result = items[0:2]
    for e in range(len(items)-2):
        sublist = sorted(items[e:e+3])
        result.append(sublist[1])
    return result


def scylla_or_charybdis(sequence, n):
    moves = {}
    correct = {}
    result = {}
    for k in range(1, len(sequence)):
        correct[k] = False
        moves[k] = 0 
        count = 0
        for e in range(k-1, len(sequence), k):
            if sequence[e] == '+':
                count += 1
                moves[k] += 1
                if count == n or count == -n:
                    correct[k] = True
                    break
            elif sequence[e] == '-':
                count -= 1
                moves[k] += 1
                if count == n or count == -n:
                    correct[k] = True
                    break
    for u in correct:
        if correct[u] == True:
            result[u] = (moves[u])
    key_min = min(result.keys(), key=(lambda i: result[i]))
    return key_min


#def is_zigzag(n):#mistake made here was not doing trial of elimination. started with finding correct cases instead of the eliminating failing cases first
#    l = str(n)
#    count = 0
#    if len(l) == 1 or len(l) == 2 and int(l[0]) != int(l[1]):
#        return True
 #   elif len(l) == 4 and int(l[0]) < int(l[1]) > int(l[2]) < int(l[3]) or int(l[0]) > int(l[1]) < int(l[2]) > int(l[3]):
#        return True
#    elif int(l[0]) > int(l[1]):
 #       for e in range(1, len(l)-1, 2):
#            if int(l[e-1]) > int(l[e]) < int(l[e+1]): 
#                count += 1
#    elif int(l[0]) < int(l[1]):
#        for e in range(1, len(l)-1, 2):
#           if int(l[e-1]) < int(l[e]) > int(l[e+1]):
#                count += 1
 #   if count == (len(l)//2):
 #       return True
 #   return False


def is_zigzag(n):
    l = str(n)
    if len(l) == 1: 
        return True
    elif len(l) == 2 and int(l[0]) == int(l[1]):
        return False
    else:
        for e in range(1, len(l)-1):
            if (int(l[e-1]) >= int(l[e]) >= int(l[e+1])) or (int(l[e-1]) <= int(l[e]) <= int(l[e+1])):
                return False
    return True
    
def tukeys_ninthers(items):
    li = items
    result = []
    while len(li) > 3:
        for e in range(0, len(li)-2, 3):
            sublist = sorted(li[e:e+3])
            result.append(sublist[1])
        li = result
        if len(li) == 3:
            return sorted(li)[1]
        else:
            result = []
    return sorted(li)[1]
    
def squares_intersect(s1, s2):
    x1, y1, r1 = s1[0], s1[1], s1[2]
    x2, y2, r2 = s2[0], s2[1], s2[2]
    if x1 + r1 < x2 or y1 + r1 < y2:
        return False
    if x2 + r2 < x1 or y2 + r2 < y1:
        return False
    return True    

        
def remove_after_kth(items, k = 1):
    count = {}
    result = []
    if k == 0: 
        return []
    for value in items:
        if value in count:
            count[value] += 1
        else:
            count[value] = 1
        if count[value] <= k:
            result.append(value)
    return result
            
def josephus(n, k):
    
    res, soldiers = [], list(range(1, n+1))
    
    while n > 0:
        pos = (k-1) % n
        res.append(soldiers[pos])
        
        if pos == n-1:# is element the last element in the list?
            del(soldiers[-1])#soldiers = soldiers[:-1]
        else:
            soldiers = soldiers[pos+1:] + soldiers[:pos]
        n -= 1
    return res

def autocorrect_word(word, words, df):
    best = ""
    bd = 10000
    for w in words:
        if len(w) == len(word):
            d = 0 #calculate distance bw words
            for c1, c2 in zip(w, word):
                d += df(c1, c2) # sum of differences between the characters (all characters)
            #d = sum([df(c1,c2)] for c1, c2 in zip(w, word)])#no change in efficiency, but this is more pythonic code styling
            if d < bd:
                bd = d
                best = w
    return best
    
def brangelina (first, second):
    
    i = 0 
    while second[i] not in "aeiou":
        i += 1
    second = second[i:]
    
    groups = []
    in_group = False
    for i, c in enumerate(first):
        if c in "aeiou":
            if not in_group:
                in_group = True #if in vowel, append position of vowel
                groups.append(i)
        else:
            in_group = False
    
    if (len(groups) == 1):
        first = first[:groups[0]]
    else:
        first = first[:groups[-2]]
        
    return first + second
    
def first_preceded_by_smaller(items, k = 1):#creating new lists is expensive
    dominators = {}
    l1 = items[:k]
    for index, value in enumerate(items[k:]):
        dominators[index] = 0
        for e in range(len(l1)):
            if items[e] < value:
                dominators[index] += 1
        if dominators[index] >= k:
            return value
            break
        l1 = items[:k+(index+1)]

def crag_score(dice):
    sort = sorted(dice)
    a, b, c = sort[0], sort[1], sort[2]
    if sum(dice) == 13:
        if a == b or b == c or c == a:
            return 50
        else:
            return 26
    elif a == b == c:
        return 25
    elif a == 1 and b == 2 and c == 3:
        return 20
    elif a == 4 and b == 5 and c == 6:
        return 20
    elif a == 1 and b == 3 and c == 5:
        return 20
    elif a == 2 and b == 4 and c == 6:
        return 20
    elif a + b <= c:
        return c
    elif a + b > c:
        if a == b:
            return a + b
        elif b == c:
            return b + c
        else:
            return c

def count_distinct_sums_and_products(items):
    distinct = set()
    i = list(it.combinations(items + items, 2))
    for pair in range(len(i)):
        sums = i[pair][0] + i[pair][1]
        prod = i[pair][0] * i[pair][1]
        distinct.add(sums)
        distinct.add(prod)
    return len(distinct)    
    
def safe_squares_rooks(n, rooks):
    row = n
    col = n
    urow = set()
    ucol = set()
    for each in rooks:
        urow.add(each[0])
        ucol.add(each[1])
    return (row - len(urow))*(col - len(ucol))

def safe_squares_bishops(n, bishops):
    board = list(it.product(range(0, n), repeat = 2))
    notsafe = set()
    for bishop in bishops:
        for square in board:
            if abs(bishop[0]-square[0]) == abs(bishop[1]-square[1]):
                notsafe.add(square)
    return abs(len(board)-len(notsafe))

def count_growlers(animals):
    growl = 0
    r = {'dog':0, 'cat':0, 'god':0, 'tac':0}
    l = {'dog':0, 'cat':0, 'god':0, 'tac':0}
    for animal in animals:
        r[animal] += 1
    for animal in animals:
        r[animal] -= 1
        if animal in ('god', 'tac'):
            if (r['dog'] + r['god']) - (r['cat'] + r['tac']) > 0:
                growl += 1
        else:
            if (l['dog'] + l['god']) - (l['cat'] + l['tac']) > 0:
                growl += 1
        l[animal] += 1
        
    #slow solution, creation of 2 new lists per iteration is expensive
    #for index, animal in enumerate(animals):
    #    right = animals[index+1:]
    #    left = animals[:index]
    #    if animal in ('god', 'tac'):#facing right
    #        if (right.count('god') + right.count('dog')) - (right.count('cat') + right.count('tac')) > 0:
    #           growl += 1
    #    else: #if animal in ('dog','cat) facing left
    #        if (left.count('god') + left.count('dog')) - (left.count('cat') + left.count('tac')) > 0:
    #            growl += 1
    return growl
 
def double_trouble(items, n):
    #the list doubles every a times
    if len(items) == 1 or n == 1:
        return items[0]
    
    a = len(items) * 2
    b = len(items)
    count = 2
    d = {}
    l = co.deque(items)
    x = 0
    if n < 100:
        for times in range(n):
            l.append(l[0])
            l.append(l[0])
            x = l[0]
            l.popleft()
        return x
        
    while b <= n:
        b += a
        a *= 2
        d[count] = b
        count += 1 
        
    split = abs(d[list(d)[-1]] - d[list(d)[-2]]) // len(items)
    build = d[list(d)[-2]]
    
    for i in range(1, len(items)+1):
        if build < n < build + split*i:
            return items[i-1]
    return items[-1]
    
def give_change(amount, coins):
    change = []
    for coin in coins:
        while amount >= coin:
            amount = amount - coin
            change.append(coin)
    return change
            
def nearest_polygonal_number(n,s):
    #binary search
    begin = 1
    end = 2
    formula = 0
    if n == 1:
        return 1
    
    while n > formula: #finding a sufficiently large upperbound to capture i
        end = end**2
        formula = ((s - 2) * end**2 - (s - 4) * end) // 2
    
    while end - begin >= 2: #binary search
        midpoint = (begin + end) // 2
        midpoint_value = ((s - 2) * midpoint**2 - (s - 4) * midpoint) // 2
        if midpoint_value == n:
            return midpoint
        elif n < midpoint_value:
            end = midpoint 
        else:
            begin = midpoint 
    
    #begin, end, which has smaller diff
    end_value = (((s - 2) * end**2 - (s - 4) * end) // 2)
    begin_value = (((s - 2) * begin**2 - (s - 4) * begin) // 2)
    if abs(end_value - n) < abs(begin_value - n):
        return end_value
    else:
        return begin_value         

def unscramble(words, word):
    
    alphabetw = dict.fromkeys(string.ascii_lowercase, 0)
    alphabetword = dict.fromkeys(string.ascii_lowercase, 0)
    result = []
    for letter in word:
        alphabetword[letter] += word.count(letter) 
        
    for w in words:
        if len(w) == len(word) and word[0] == w[0] and word[-1] == w[-1]:
            for letter in w:
                alphabetw[letter] += w.count(letter)
            if alphabetword == alphabetw:
                result.append(w)
            alphabetw = alphabetw.fromkeys(alphabetw, 0)
    return result
    
def recaman(n):
    used = set()
    seq = [0]
    for i in range(1, n+1):
        if seq[i - 1] - i > 0 and seq[i - 1] - i not in used:
            seq.append(seq[i - 1] - i)
            used.add(seq[i - 1] - i)
        else:
            seq.append(seq[i - 1] + i)
            used.add(seq[i - 1] + i)
    return seq[1:]
    
def bridge_hand_shape(hand):
    suit = {'spades':0, 'hearts':0, 'diamonds':0, 'clubs':0}
    for card in hand:
        if card[1] in suit:
            suit[card[1]] += 1
    return list(suit.values())
    
def count_divisibles_in_range(start, end, n):   
    if abs(start) == abs(end):
        return (end // n) * 2 + 1
    elif start > 0 and end > 0:
        if start % n == 0:
            return ((end // n) - (start // n)) + 1
        else: return (end // n) - (start // n)
    elif start == 0 and end > 0:
        return end // n + 1
    elif start < 0 and end == 0:
        return abs(start) // n + 1
    elif start < 0 and end < 0:
        return abs(start - end) // n
    elif start < 0 and end > 0:
        return (abs(start) + end) // n
        
def milton_work_point_count(hand, trump = 'notrump'):
    total = 0
    ranks = {'ace':0, 'king':0, 'queen':0, 'jack':0}
    suit = {'spades':0, 'hearts':0, 'diamonds':0, 'clubs':0}
    flat = [[4,3,3,3], [3,4,3,3], [3,3,4,3], [3,3,3,4]]
    
    for card in hand:
        if card[1] in suit:
            suit[card[1]] += 1
        
    for card in hand:
        if card[0] in ranks:
            ranks[card[0]] += 1
            
    total += ((ranks['ace'] * 4) + (ranks['king'] * 3) + (ranks['queen'] * 2) + ranks['jack'])
    
    if list(suit.values()) in flat:
        total -= 1
       
    for each in suit:
        if suit[each] == 5:
            total += 1
        elif suit[each] == 6:
            total += 2
        elif suit[each] >= 7:
            total += 3
    
    if trump != 'notrump':
        for each in suit:
            if suit[each] == 0 and each != trump:
                total += 5
            elif suit[each] == 1 and each != trump:
                total += 3
    
    return total
    
def aliquot_sequence(n, giveup = 100):
    aliquotnumbers = [n]
    divisors = []
    for e in range(giveup-1):
        for i in range(1, aliquotnumbers[-1]):
            if aliquotnumbers[-1] % i == 0:
                divisors.append(i)
        if aliquotnumbers[-1] != sum(divisors):
            aliquotnumbers.append(sum(divisors))
        divisors = []
        if aliquotnumbers[-1] == 0:
            break
    return aliquotnumbers

lattices = {}
def lattice_paths(x, y, tabu):
    if x == 0 or y == 0:
        lattices[(x, y)] = 1
    if (x, y) in tabu:
        lattices[(x, y)] = 0
    if x < 0 or y < 0:
        lattices[(x, y)] = 0    
    if (x, y) in lattices:
        return lattices[(x, y)]
    else:
        lattice = lattice_paths(x, y - 1, tabu) + lattice_paths(x - 1, y, tabu)
        lattices[(x, y)] = lattice
        return lattice

