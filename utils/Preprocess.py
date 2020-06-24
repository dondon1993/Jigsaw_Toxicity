
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import string


# In[2]:


puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']',
          '>', '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^',
          '®', '`', '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█',
          '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶',
          '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼',
          '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲',
          'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪',
          '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']

star_dict = [
    ("sh*t", "shit"),
    ("s**t", "shit"),
    ("f*ck", "fuck"),
    ("fu*k", "fuck"),
    ("f**k", "fuck"),
    ("f*****g", "fucking"),
    ("f***ing", "fucking"),
    ("f**king", "fucking"),
    ("p*ssy", "pussy"),
    ("p***y", "pussy"),
    ("pu**y", "pussy"),
    ("p*ss", "piss"),
    ("b*tch", "bitch"),
    ("bit*h", "bitch"),
    ("h*ll", "hell"),
    ("h**l", "hell"),
    ("cr*p", "crap"),
    ("d*mn", "damn"),
    ("stu*pid", "stupid"),
    ("st*pid", "stupid"),
    ("n*gger", "nigger"),
    ("n***ga", "nigger"),
    ("f*ggot", "faggot"),
    ("scr*w", "screw"),
    ("pr*ck", "prick"),
    ("g*d", "god"),
    ("s*x", "sex"),
    ("a*s", "ass"),
    ("a**hole", "asshole"),
    ("a***ole", "asshole"),
    ("a**", "ass"),
]

star_replacer = [
    (re.compile(pat.replace("*", "\*"), flags=re.IGNORECASE), repl)
    for pat, repl in star_dict
]

misspell_dict = {"aren't": "are not", "can't": "can not", "couldn't": "could not", 'ppl': 'people', 'nyt': 'newyork time',
                 "didn't": "did not", "doesn't": "does not", "don't": "do not", 'nytime': 'newyork time',
                 "hadn't": "had not", "hasn't": "has not", "haven't": "have not", ' bc ': ' becasue ', '\n':' ',
                 "he'd": "he would", "he'll": "he will", "he's": "he is",
                 "i'd": "i had", "i'll": "i will", "i'm": "i am", "isn't": "is not",
                 "it's": "it is", "it'll": "it will", "i've": "i have", "let's": "let us",
                 "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
                 "she'd": "she would", "she'll": "she will", "she's": "she is",
                 "shouldn't": "should not", "that's": "that is", "there's": "there is",
                 "they'd": "they would", "they'll": "they will", "they're": "they are",
                 "they've": "they have", "we'd": "we would", "we're": "we are",
                 "weren't": "were not", "we've": "we have", "what'll": "what will",
                 "what're": "what are", "what's": "what is", "what've": "what have",
                 "where's": "where is", "who'd": "who would", "who'll": "who will",
                 "who're": "who are", "who's": "who is", "who've": "who have",
                 "won't": "will not", "wouldn't": "would not", "you'd": "you would",
                 "you'll": "you will", "you're": "you are", "you've": "you have",
                 "'re": " are", "wasn't": "was not", "we'll": "we will", "tryin'": "trying",
                "\x89Û_".lower(): "", "\x89ÛÒ".lower(): "", "\x89ÛÓ".lower(): "", "\x89ÛÏWhen".lower(): "when", "\x89ÛÏ".lower(): "",
                "china\x89Ûªs".lower(): "china's", "let\x89Ûªs".lower(): "let's", "\x89Û÷".lower(): "", "\x89Ûª": "",
                "\x89Û\x9d".lower(): "", "å_": "", "\x89Û¢".lower(): "", "\x89Û¢åÊ".lower(): "", "fromåÊwounds".lower(): "from wounds",
                "åÊ".lower(): "", "åÈ".lower(): "", "JapÌ_n".lower(): "Japan".lower(), "Ì©".lower(): "e", "å¨": "", "SuruÌ¤".lower(): "Suruc".lower(), "åÇ".lower(): "",
                "å£3million": "3 million", "åÀ".lower(): "", "don\x89Ûªt".lower(): "do not", "I\x89Ûªm".lower(): "i am", 
                "here's": "here is", "you\x89Ûªve".lower(): "you have", "it\x89Ûªs": "it is", "doesn\x89Ûªt".lower(): "does not",
                "It\x89Ûªs".lower(): "it is", "Here\x89Ûªs".lower(): "here is", "I\x89Ûªve".lower(): "i have", "y'all": "you all",
                "can\x89Ûªt".lower(): "can not", "would've": "would have", "it'll": "it will", "wouldn\x89Ûªt".lower(): "would not",
                "That\x89Ûªs".lower(): "that is", "should've": "should have", "You\x89Ûªre".lower(): "you are", "Don\x89Ûªt".lower(): "do not",
                "Can\x89Ûªt".lower(): "cannot", "you\x89Ûªll".lower(): "you will", "I\x89Ûªd".lower(): "i would", "ain't": "am not",
                "could've": "could have", "youve": "you have", "donå«t": "do not", "&gt;": "", "&lt;": "", "&amp;": "",
                "w/e": "whatever", "w/": "with", "USAgov".lower(): "usa government", "recentlu": "recently", "ph0tos": "photos",
                "amirite": "am i right", "exp0sed": "exposed", "<3": "love", "amageddon": "armageddon", "trfc": "traffic",
                "windstorm": "wind storm", "lmao": "laughing my ass off", "irandeal": "iran deal", "arianagrande": "ariana grande",
                "camilacabello97": "camila cabello", "rondarousey": "ronda rousey", "mtvhottest": "mtv hottest", "trapmusic": "trap music",
                "prophetmuhammad": "prophet muhammad", "pantherattack": "panther attack", "strategicpatience": "strategic patience",
                "socialnews": "social news", "nasahurricane": "nasa hurricane", "onlinecommunities": "online communities",
                "humanconsumption": "human consumption", "typhoon-devastated": "typhoon devastated", "meat-loving": "meat loving",
                "facialabuse": "facial abuse", "lakecounty": "lake county", "beingauthor": "being author", "withheavenly": "with heavenly",
                "thanku": "thank you", "itunesmusic": "itunes music", "offensivecontent": "offensive content", "worstsummerjob": "worst summer job",
                "harrybecareful": "harry be careful", "nasasolarsystem": "nasa solar system", "animalrescue": "animal rescue", "kurtschlichter": "kurt schlichter",
                "throwingknifes": "throwing knives", "godslove": "god's love", "bookboost": "book boost", "ibooklove": "I book love", 
                "nestleindia": "nestle india", "realdonaldtrump": "donald trump", "davidvonderhaar": "david vonderhaar", "cecilthelion": "cecil the lion",
                "weathernetwork": "weather network", "withbioterrorism&use": "with bioterrorism & use", "hostage&2": "hostage & 2",
                "gopdebate": "gop debate", "rickperry": "rick perry", "frontpage": "front page", "newsintweets": "news in tweets",
                "viralspell": "viral spell", "til_now": "until now", "volcanoinrussia": "volcano in russia", "zippednews": "zipped news",
                "michelebachman": "michele bachman", "53inch": "53 inch", "kerricktrial": "kerrick trial", "abstorm": "alberta storm",
                "beyhive": "beyonce hive", "idfire": "idaho fire", "detectado": "detected", "rockyfire": "rocky fire",
                "listen/buy": "listen / buy", "nickcannon": "nick cannon", "faroeislands": "faroe islands", "yycstorm": "calgary storm",
                "idps:": "internally displaced people:", "artistsunited": "artists united", "claytonbryant": "clayton bryant",
                "jimmyfallon": "jimmy fallon", "justinbieber": "justin bieber", "utc2015": "utc 2015", "time2015": "time 2015",
                "djicemoon": "dj icemoon", "livingsafely": "living safely", "fifa16": "fifa 2016", "thisiswhywecanthavenicethings": "this is why we can not have nice things",
                "bbcnews": "bbc news", "undergroundrailraod": "underground railraod", "c4news": "c4 news", "nosurrender": "no surrender",
                "notexplained": "not explained", "greatbritishbakeoff": "great british bake off", "londonfire": "london fire",
                "kotaweather": "kota weather", "luchaunderground": "lucha underground", "koin6news": "koin 6 news", "liveonk2": "live on k2",
                "nikeplus": "nike plus", "9newsgoldcoast": "9 news gold coast", "david_cameron": "david cameron",
                "peterjukes": "peter jukes", "jamesmelville": "james melville", "megynkelly": "megyn kelly", "cnewslive": "c news live",
                "jamaicaobserver": "jamaica observer", "tweetlikeitsseptember11th2001": "tweet like it is september 11th 2001",
                "cbplawyers": "cbp lawyers", "fewmoretweets": "few more tweets", "blacklivesmatter": "black lives matter",
                "cjoyner": "chris joyner", "engvaus": "england vs australia", "scottwalker": "scott walker", "mikeparractor": "michael parr",
                "4playthursdays": "foreplay thursdays", "tgf2015": "tontitown grape festival", "realmandyrain": "mandy rain", "graysondolan": "grayson dolan",
                "apollobrown": "apollo brown", "saddlebrooke": "saddlebrooke", "tontitowngrape": "tontitown grape", "abbswinston": "abbs winston",
                "shaunking": "shaun king", "meekmill": "meek mill", "tornadogiveaway": "tornado giveaway", "grupdates": "gr updates",
                "southdowns": "south downs", "braininjury": "brain injury", "auspol": "australian politics", "plannedparenthood": "planned parenthood",
                "calgaryweather": "calgary weather", "weallheartonedirection": "we all heart one direction", "edsheeran": "ed sheeran",
                "trueheroes": "true heroes", "s3xleak": "sex leak", "complexmag": "complex magazine", "theadvocatemag": "the advocate magazine",
                "cityofcalgary": "city of calgary", "ebolaoutbreak": "ebola outbreak", "summerfate": "summer fate", "ramag": "royal academy magazine",
                "offers2go": "offers to go", "foodscare": "food scare", "MNPDNashville".lower(): "metropolitan nashville police department",
                "tflbusalerts": "tfl bus alerts", "gamergate": "gamer gate", "ihhen": "humanitarian relief", "spinningbot": "spinning bot",
                "modiministry": "modi ministry", "TAXIWAYS".lower(): "taxi ways", "Calum5SOS".lower(): "calum hood", "po_st": "po.st",
                "scoopit": "scoop.it", "ultimalucha": "ultima lucha", "jonathanferrell": "jonathan ferrell", "aria_ahrary": "aria ahrary",
                "rapidcity": "rapid city", "lavenderpoetrycafe": "lavender poetry cafe", "eudrylantiqua": "eudry lantiqua",
                "15pm": "15 pm", "originalfunko": "funko", "rightwaystan": "richard tan", "cindynoonan": "cindy noonan", "rt_america": "rt america",
                "narendramodi": "narendra modi", "bakeofffriends": "bake off friends", "teamhendrick": "hendrick motorsports",
                "alexbelloli": "alex belloli", "itsjustinstuart": "justin stuart", "gunsense": "gun sense", "debatequestionswewanttohear": "debate questions we want to hear",
                "royalcarribean": "royal carribean", "samanthaturne19": "samantha turner", "jonvoyage": "jon stewart",
                "renew911health": "renew 911 health","suryaray": "surya ray", "pattonoswalt": "patton oswalt", "minhazmerchant": "minhaz merchant",
                "tlvfaces": "israel diaspora coalition", "pmarca": "marc andreessen", "pdx911": "portland police", "jamaicaplain": "jamaica plain",
                "japton": "arkansas", "routecomplex": "route complex", "insubcontinent": "indian subcontinent", "njturnpike": "new jersey turnpike",
                "politifiact": "politifact", "hiroshima70": "hiroshima", "gmmbc": "greater mt moriah baptist church",
                "versethe": "verse the", "tubestrike": "tube strike", "missionhills": "mission hills", "protectdenaliwolves": "protect denali wolves",
                "NANKANA".lower(): "nankana", "newz_sacramento": "news sacramento", "gofundme": "go fund me", "pmharper": "stephen harper",
                "ivanberroa": "ivan berroa", "losdelsonido": "los del sonido", "bancodeseries": "banco de series", "timkaine": "tim kaine",
                "identitytheft": "identity theft", "alllivesmatter": "all lives matter", "mishacollins": "misha collins", "BillNeelyNBC".lower(): "bill neely",
                "beclearoncancer": "be clear on cancer", "kowing": "knowing", "screamqueens": "scream queens", "askcharley": "ask charley",
                "blizzheroes": "heroes of the storm", "bradleybrad47": "bradley brad", "hannaph": "typhoon hanna", "meinlcymbals": "meinl cymbals",
                "ptbo": "peterborough", "cnnbrk": "cnn breaking news", "indiannews": "indian news", "savebees": "save bees",
                "greenharvard": "green harvard", "standwithpp": "stand with planned parenthood", "hermancranston": "herman cranston",
                "wmur9": "wmur tv", "RockBottomRadFM".lower(): "rock bottom radio", "ameenshaikh3": "ameen shaikh", "prosyn": "project syndicate",
                "daesh": "isis", "s2g": "swear to god", "listenlive": "listen live", "cdcgov": "centers for disease control and prevention",
                "foxnew": "fox news", "cbsbigbrother": "big brother", "juliedicaro": "julie dicaro", "theadvocatemag": "the advocate magazine",
                "RohnertParkDPS".lower(): "rohnert park police department", "THISIZBWRIGHT".lower(): "bonnie wright", "wildhorses": "wild horses",
                "fantasticfour": "fantastic four", "bathandnortheastsomerset": "bath and northeast somerset", "thatswhatfriendsarefor": "that is what friends are for",
                "residualincome": "residual income", "yahoonewsdigest": "yahoo news digest", "malaysiaairlines": "malaysia airlines",
                "amazondeals": "amazon deals", "misscharleywebb": "charley webb", "shoalstraffic": "shoals traffic", "georgefoster72": "george foster",
                "pop2015": "pop 2015", "_pokemoncards_": "pokemon cards", "dianneg": "dianne gallagher", "kashmirconflict": "kashmir conflict", 
                "britishbakeoff": "british bake off", "freekashmir": "free kashmir", "mattmosley": "matt mosley", "bishopfred": "bishop fred",
                "endconflict": "end conflict", "endoccupation": "end occupation", "charlesdagnall": "charles dagnall", "latestnews": "latest news",
                "kindlecountdown": "kindle countdown", "nomorehandouts": "no more handouts", "datingtips": "dating tips", "charlesadler": "charles adler",
                "twia": "texas windstorm insurance association", "txlege": "texas legislature", "windstorminsurer": "windstorm insurer",
                "newss": "news", "hempoil": "hemp oil", "commoditiesare": "commodities are", "tubestrike": "tube strike", "joenbc": "joe scarborough",
                "literarycakes": "literary cakes", "ti5": "the international 5", "thehill": "the hill", "3others": "3 others", "stighefootball": "sam tighe",
                "whatstheimportantvideo": "what is the important video", "claudiomeloni": "claudio meloni", "dukeskywalker": "duke skywalker",
                "carsonmwr": "fort carson", "offdishduty": "off dish duty", "andword": "and word", "rhodeisland": "rhode island", "easternoregon": "eastern oregon",
                "wawildfire": "washington wildfire", "fingerrockfire": "finger rock fire", "57am": "57 am", "jacobhoggard": "jacob hoggard",
                "newnewnew": "new new new", "under50": "under 50", "getitbeforeitsgone": "get it before it is gone", "freshoutofthebox": "fresh out of the box",
                "amwriting": "am writing", "bokoharm": "boko haram", "nowlike": "now like", "seasonfrom": "season from", "epicente": "epicenter", "epicenterr": "epicenter",
                "sicklife": "sick life", "yycweather": "calgary weather", "calgarysun": "calgary sun", "approachng": "approaching", "evng": "evening",
                "sumthng": "something", "ellenpompeo": "ellen pompeo", "shondarhimes": "shonda rhimes", "abcnetwork": "abc network",
                "sushmaswaraj": "sushma swaraj", "pray4japan": "pray for japan", "hope4japan": "hope for japan", "illusionimagess": "illusion images",
                "summerunderthestars": "summer under the stars", "shallwedance": "shall we dance", "tcmparty": "tcm party", "marijuananews": "marijuana news", 
                "onbeingwithkristatippett": "on being with krista tippett", "beingtweets": "being tweets", "newauthors": "new authors", "remedyyyy": "remedy", "44pm": "44 pm",
                "headlinesapp": "headlines app", "40pm": "40 pm", "myswc": "severe weather center", "ithats": "that is", 
                "icouldsitinthismomentforever": "I could sit in this moment forever", "fatloss": "fat loss", "02pm": "02 pm", "metrofmtalk": "metro fm talk",
                "bstrd": "bastard", "bldy": "bloody", "terrorismturn": "terrorism turn", "bbcnewsasia": "bbc news asia",
                "behindthescenes": "behind the scenes", "georgetakei": "george takei", "womensweeklymag": "womens weekly magazine", 
                "survivorsguidetoearth": "survivors guide to earth", "incubusband": "incubus band", "babypicturethis": "baby picture this",
                "bombeffects": "bomb effects", "win10": "windows 10", "idkidk": "I do not know I do not know", "thewalkingdead": "the walking dead",
                "amyschumer": "amy schumer", "crewlist": "crew list", "erdogans": "erdogan", "bbclive": "bbc live", "TonyAbbottMHR".lower(): "tony abbott",
                "paulmyerscough": "paul myerscough", "georgegallagher": "george gallagher", "jimmiejohnson": "jimmie johnson",
                "pctool": "pc tool", "doinghashtagsright": "doing hashtags right", "throwbackthursday": "throwback thursday", "snowbacksunday": "snowback sunday",
                "lakeeffect": "lake effect", "rtphotographyuk": "richard thomas photography uk", "bigbang_cbs": "big bang cbs",
                "writerslife": "writers life", "naturalbirth": "natural birth", "unusualwords": "unusual words", "wizkhalifa": "wiz khalifa",
                "acreativedc": "a creative dc", "vscodc": "vsco dc", "vscocam": "vsco camera", "thebeachdc": "the beach dc", "buildingmuseum": "building museum",
                "worldoil": "world oil", "redwedding": "red wedding", "amazingracecanada": "amazing race canada", "wakeupamerica": "wake up america",
                "\\allahuakbar\\": "allahu akbar", "bleased": "blessed", "nigeriantribune": "nigerian tribune", "HIDEO_KOJIMA_EN".lower(): "hideo kojima",
                "fusionfestival": "fusion festival", "50mixed": "50 mixed", "noagenda": "no agenda", "whitegenocide": "white genocide",
                "dirtylying": "dirty lying", "syrianrefugees": "syrian refugees", "changetheworld": "change the world", "ebolacase": "ebola case",
                "mcgtech": "mcg technologies", "withweapons": "with weapons", "advancedwarfare": "advanced warfare", "letsfootball": "let us football",
                "latenitemix": "late night mix", "philcollinsfeed": "phil collins", "rudyhavenstein": "rudy havenstein", "22pm": "22 pm",
                "54am": "54 am", "38am": "38 am", "oldfolkexplainstuff": "old folk explain stuff", "blacklivesmatter": "black lives matter",
                "insanelimits": "insane limits", "youcantsitwithus": "you cannot sit with us", "2k15": "2015", "theiran": "iran",
                "jimmyfallon": "jimmy fallon", "albertbrooks": "albert brooks", "defense_news": "defense news", "nuclearrcsa": "nuclear risk control self assessment",
                "auspol": "australia politics", "nuclearpower": "nuclear power", "whiteterrorism": "white terrorism", "truthfrequencyradio": "truth frequency radio",
                "erasureisnotequality": "erasure is not equality", "probononews": "pro bono news", "jakartapost": "jakarta post",
                "toopainful": "too painful", "melindahaunton": "melinda haunton", "nonukes": "no nukes", "curryspcworld": "currys pc world",
                "ineedcake": "i need cake", "blackforestgateau": "black forest gateau", "bbcone": "bbc one", "alexxpage": "alex page",
                "jonathanserrie": "jonathan serrie", "socialjerkblog": "social jerk blog", "chelseavperetti": "chelsea peretti",
                "irongiant": "iron giant", "ronfunches": "ron funches", "timcook": "tim cook", "sebastianstanisaliveandwell": "sebastian stan is alive and well",
                "madsummer": "mad summer", "nowyouknow": "now you know", "concertphotography": "concert photography", "tomlandry": "tom landry", 
                "showgirldayoff": "show girl day off", "yougslavia": "yugoslavia", "quantumdatainformatics": "quantum data informatics",
                "fromthedesk": "from the desk", "theatertrial": "theater trial", "catoinstitute": "cato institute", "emekagift": "emeka gift",
                "letsbe_rational": "let us be rational", "cynicalreality": "cynical reality", "fredolsencruise": "fred olsen cruise", "notsorry": "not sorry",
                "useyourwords": "use your words", "wordoftheday": "word of the day", "dictionarycom": "dictionary.com", "thebrooklynlife": "the brooklyn life",
                "jokethey": "joke they", "nflweek1picks": "nfl week 1 picks", "uiseful": "useful", "justicedotorg": "the american association for Justice",
                "autoaccidents": "auto accidents", "stevegursten": "steve gursten", "michiganautolaw": "michigan auto law", "birdgang": "bird gang", 
                "nflnetwork": "nfl network", "nydnsports": "ny daily news sports", "RVacchianoNYDN".lower(): "ralph vacchiano ny daily news",
                "edmontonesks": "edmonton eskimos", "david_brelsford": "david brelsford", "toi_india": "the times of india", "hegot": "he got",
                "skinson9": "skins on 9", "sothathappened": "so that happened", "LCOutOfDoors".lower(): "lc out of doors", "nationfirst": "nation first",
                "indiatoday": "india today", "HLPS".lower(): "helps", "HOSTAGESTHROSW".lower(): "hostages throw", "SNCTIONS".lower(): "sanctions",
                "bidtime": "bid time", "crunchysensible": "crunchy sensible", "randomactsofromance": "random acts of romance", "momentsathill": "moments at hill",
                "eatshit": "eat shit", "liveleakfun": "live leak fun", "sahelnews": "sahel news", "abc7newsbayarea": "ABC 7 News Bay Area".lower(), "facilitiesmanagement": "facilities management",
                "facilitydude": "facility dude", "camplogistics": "camp logistics", "alaskapublic": "alaska public", "marketresearch": "market research",
                "accuracyesports": "accuracy esports", "thebodyshopaust": "the body shop australia", "yychail": "calgary hail", "yyctraffic": "calgary traffic",
                "eliotschool": "eliot school", "thebrokencity": "the broken city", "oldsfiredept": "olds fire department", "rivercomplex": "river complex", "fieldworksmells": "field work smells",
                "iranelection": "iran election", "glowng": "glowing", "kindlng": "kindling", "riggd": "rigged", "slownewsday": "slow news day",
                "myanmarflood": "myanmar flood", "abc7chicago": "ABC 7 Chicago".lower(), "copolitics": "colorado politics", "adilghumro": "adil ghumro",
                "netbots": "net bots", "byebyeroad": "bye bye road", "massiveflooding": "massive flooding", "endofus": "end of united states",
                "35pm": "35 pm", "greektheatrela": "greek theatre los angeles", "76mins": "76 minutes", "publicsafetyfirst": "public safety first", "livesmatter": "lives matter",
                "myhometown": "my hometown", "tankerfire": "tanker fire", "MEMORIALDAY".lower(): "memorial day", "MEMORIAL_DAY".lower(): "memorial day", "instaxbooty": "instagram booty",
                "jerusalem_post": "jerusalem post", "waynerooney_ina": "wayne rooney", "virtualreality": "virtual reality", "oculusrift": "oculus rift",
                "owenjones84": "owen jones", "jeremycorbyn": "jeremy corbyn", "paulrogers002": "paul rogers", "mortalkombatx": "mortal kombat x",
                "mortalkombat": "mortal kombat", "filipecoelho92": "filipe coelho", "onlyquakenews": "only quake news", "kostumes": "costumes",
                "YEEESSSS".lower(): "yes", "toshikazukatayama": "toshikazu katayama", "intldevelopment": "intl development", "extremeweather": "extreme weather",
                "werenotgrubervoters": "we are not gruber voters", "newsthousands": "news thousands", "edmundadamus": "edmund adamus", "eyewitnesswv": "eye witness wv",
                "philadelphiamuseu": "philadelphia museum", "dublincomiccon": "dublin comic con", "nicholasbrendon": "nicholas brendon",
                "alltheway80s": "all the way 80s", "fromthefield": "from the field", "northiowa": "north iowa", "willowfire": "willow fire",
                "madrivercomplex": "mad river complex", "feelingmanly": "feeling manly", "stillnotoverit": "still not over it", "fortitudevalley": "fortitude valley",
                "coastpowerlinetramtr": "coast powerline", "servicesgold": "services gold", "newsbrokenemergency": "news broken emergency", "evaucation": "evacuation",
                "leaveevacuateexitbe": "leave evacuate exit be", "P_EOPLE".lower(): "people", "tubestrike": "tube strike", "CLASS_SICK".lower(): "class sick",
                "localplumber": "local plumber", "awesomejobsiri": "awesome job siri", "payforithow": "pay for it how", "thisisafrica": "this is africa",
                "crimeairnetwork": "crime air network", "kimacheson": "kim acheson", "cityofcalgary": "city of calgary", "prosyndicate": "pro syndicate",
                "660NEWS".lower(): "660 news", "businsmagazine": "business insurance magazine", "wfocus": "focus", "shastadam": "shasta dam",
                "go2markfranco": "mark franco", "stephghinojosa": "steph hinojosa", "nashgrier": "nash grier", "nashnewvideo": "nash new video", "iwouldntgetelectedbecause": "i would not get elected because",
                "shgames": "sledgehammer games", "bedhair": "bed hair", "joelheyman": "joel heyman", "viayoutube": "via youtube",
                "$" : " dollar ","€" : " euro "," 4ao " : " for adults only "," a.m " : " before midday "," a3 " : "anytime anywhere anyplace"," aamof " : " as a matter of fact ",
                " acct " : " account "," adih " : " another day in hell "," afaic " : " as far as i am concerned "," afaict " : " as far as i can tell "," afaik " : " as far as i know ",
                " afair " : " as far as i remember "," afk " : " away from keyboard "," app " : " application "," approx " : " approximately "," apps " : " applications ",
                " asap " : " as soon as possible "," asl " : " age, sex, location "," atk " : " at the keyboard "," ave. " : " avenue ",
                " aymm " : " are you my mother "," ayor " : " at your own risk ", " b&b " : " bed and breakfast "," b+b " : " bed and breakfast ", " b.c " : " before christ ",
                " b2b " : " business to business "," b2c " : " business to customer "," b4 " : " before "," b4n " : " bye for now "," b@u " : " back at you ",
                " bae " : " before anyone else "," bak " : " back at keyboard "," bbbg " : " bye bye be good "," bbc " : " british broadcasting corporation ",
                " bbias " : " be back in a second "," bbl " : " be back later "," bbs " : " be back soon "," be4 " : " before "," bfn " : " bye for now ",
                " blvd " : " boulevard "," bout " : " about "," brb " : " be right back "," bros " : " brothers "," brt " : " be right there "," bsaaw ": " big smile and a wink ",
                " btw ": " by the way "," bwl ": " bursting with laughter "," c/o ": " care of "," cet " : " central european time "," cf " : " compare ",
                " cia " : " central intelligence agency "," csl " : " can not stop laughing "," cu " : " see you "," cul8r " : " see you later "," cv " : " curriculum vitae ",
                " cwot " : " complete waste of time "," cya " : " see you "," cyt " : " see you tomorrow "," dae " : " does anyone else ",
                " dbmib " : " do not bother me i am busy "," diy " : " do it yourself "," dm " : " direct message "," dwh " : " during work hours ",
                " e123 " : " easy as one two three "," eet " : " eastern european time "," eg " : " example "," embm " : " early morning business meeting ",
                " encl " : " enclosed "," encl. " : " enclosed "," etc " : " and so on "," faq " : " frequently asked questions "," fawc " : " for anyone who cares ",
                " fb " : " facebook "," fc " : " fingers crossed "," fig " : " figure "," fimh " : " forever in my heart "," ft. " : " feet ",
                " ft " : " featuring "," ftl " : " for the loss "," ftw " : " for the win "," fwiw " : " for what it is worth "," fyi " : " for your information ",
                " g9 " : " genius "," gahoy " : " get a hold of yourself "," gal " : " get a life "," gcse " : " general certificate of secondary education ",
                " gfn " : " gone for now "," gg " : " good game "," gl " : " good luck "," glhf " : " good luck have fun "," gmt " : " greenwich mean time ",
                " gmta " : " great minds think alike "," gn " : " good night "," g.o.a.t " : " greatest of all time "," goat ":" greatest of all time ",
                " goi " : " get over it "," gps " : " global positioning system "," gr8 " : " great "," gratz " : " congratulations ",
                " gyal " : " girl "," h&c " : " hot and cold "," hp " : " horsepower "," hr " : " hour "," hrh " : " his royal highness ",
                " ht " : " height "," ibrb " : " i will be right back "," ic " : " i see "," icq " : " i seek you "," icymi " : " in case you missed it ",
                " idc " : " i do not care "," idgadf " : " i do not give a damn fuck "," idgaf " : " i do not give a fuck "," idk " : " i do not know ",
                " ie " : " that is "," i.e " : " that is "," ifyp " : " i feel your pain "," IG " : " instagram "," iirc " : " if i remember correctly ",
                " ilu " : " i love you "," ily " : " i love you "," imho " : " in my humble opinion "," imo " : " in my opinion "," imu " : " i miss you ",
                " iow " : " in other words "," irl " : " in real life "," j4f " : " just for fun "," jic " : " just in case "," jk " : " just kidding ",
                " jsyk " : " just so you know "," l8r " : " later "," lb " : " pound "," lbs " : " pounds "," ldr " : " long distance relationship ",
                " lmao " : " laugh my ass off "," lmfao " : " laugh my fucking ass off "," lol " : " laughing out loud "," ltd " : " limited ",
                " ltns " : " long time no see "," m8 " : " mate "," mf " : " motherfucker "," mfs " : " motherfuckers "," mfw " : " my face when ",
                " mofo " : " motherfucker "," mph " : " miles per hour "," mr " : " mister "," mrw " : " my reaction when "," ms " : " miss ",
                " mte " : " my thoughts exactly "," nagi " : " not a good idea "," nbc " : " national broadcasting company ",
                " nbd " : " not big deal "," nfs " : " not for sale "," ngl " : " not going to lie "," nhs " : " national health service ",
                " nrn " : " no reply necessary "," nsfl " : " not safe for life "," nsfw " : " not safe for work "," nth " : " nice to have ",
                " nvr " : " never "," nyc " : " new york city "," oc " : " original content "," og " : " original ",
                " ohp " : " overhead projector "," oic " : " oh i see "," omdb " : " over my dead body "," omg " : " oh my god ",
                " omw " : " on my way "," p.a " : " per annum "," p.m " : " after midday "," pm " : " prime minister ",
                " poc " : " people of color "," pov " : " point of view "," pp " : " pages "," ppl " : " people ",
                 " prw " : " parents are watching "," ps " : " photoshop "," pt " : " point "," ptb " : " please text back ",
                " pto " : " please turn over "," qpsa " : " what happens "," ratchet " : " rude "," rbtl " : " read between the lines ",
                " rlrt " : " real life retweet ", " rofl " : " rolling on the floor laughing "," roflol " : " rolling on the floor laughing out loud ",
                " rotflmao " : " rolling on the floor laughing my ass off "," rt " : " retweet "," ruok " : " are you ok ",
                " sfw " : " safe for work "," sk8 " : " skate "," smh " : " shake my head "," sq " : " square ",
                " srsly " : " seriously "," ssdd " : " same stuff different day "," tbh " : " to be honest ",
                " tbs " : " tablespooful "," tbsp " : " tablespooful "," tfw " : " that feeling when "," thks " : " thank you ",
                " tho " : " though "," thx " : " thank you "," tia " : " thanks in advance "," til " : " today i learned ",
                " tl;dr " : " too long i did not read "," tldr " : " too long i did not read "," tmb " : " tweet me back ",
                " tntl " : " trying not to laugh "," ttyl " : " talk to you later "," u " : " you "," u2 " : " you too ",
                " u4e " : " yours for ever "," utc " : " coordinated universal time "," w/ " : " with "," w/o " : " without ",
                " w8 " : " wait "," wassup " : " what is up "," wb " : " welcome back "," wtf " : " what the fuck "," wtg " : " way to go ",
                " wtpa " : " where the party at "," wuf " : " where are you from "," wuzup " : " what is up "," wywh " : " wish you were here ",
                " yd " : " yard "," ygtr " : " you got that right "," ynk " : " you never know "," zzz " : " sleeping bored and tired "}


# In[3]:


def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)
    
def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in ['target'] + identity_columns:
        if (col == 'target') & ('target' not in df.columns):
            col = 'toxicity'
        convert_to_bool(bool_df, col)
    return bool_df


# In[4]:


def clean_tweets(row):
    """Removes links and non-ASCII characters"""
    
    row = ''.join([x for x in row if x in string.printable])
    # Removing URLs
    row = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", row)
    
    return row


# In[5]:


def replace_misspell(x):
    
    for (key, value) in misspell_dict.items():
        if key in x:
            x = x.replace(key, value)
                
    return x


# In[6]:


def replace_puncts(row):
    
    if '...' in row:
        row = row.replace('...',' ')
    
    for punct in puncts + list(string.punctuation):
        
        if punct in row:
            row = row.replace(punct, f' {punct} ')
            
    return row


# In[7]:


def replace_at(row):
    if '@' not in row:
        return row
    new_word = ''
    for word in row.split():
        if '@' in word:
            new_word += '@' + ' '
            continue
        else:
            new_word += word + ' '
            
    new_word.strip()
    return new_word


# In[8]:


def replace_numbers(row):
    return re.sub(r'\d+', ' ', row)


# In[9]:


def replace_star(row):
    for pattern, repl in star_replacer:
        row = pattern.sub(repl, row)
    return row


# In[10]:


class preproc_config:
    
    def __init__(self, lower_case, replace_at, clean_tweets, replace_misspell, replace_star, replace_puncts):
        
        self.lower_case = lower_case
        self.replace_at = replace_at
        self.clean_tweets = clean_tweets
        self.replace_misspell = replace_misspell
        self.replace_star = replace_star
        self.replace_puncts = replace_puncts


# In[1]:


def prepare_train_text(df, preproc_config):
    
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    rude_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
    
    weights = np.ones((len(df),))
    weights += df[identity_columns].fillna(0).values.sum(axis=1) * 2 # 3
    weights += df[rude_columns].fillna(0).values.sum(axis=1) * 1   # Originally doesn't exist
    weights += df['target'].values * 3.5                             # 8
    weights /= weights.max()

    df['weight'] = weights
    
    if preproc_config.lower_case == True:   
        df['text_proc'] = df['comment_text'].str.lower()
    else:
        df['text_proc'] = df['comment_text']
        
    if preproc_config.replace_at == True:
        df['text_proc'] = df['text_proc'].apply(replace_at)
    
    if preproc_config.clean_tweets == True:
        df['text_proc'] = df['text_proc'].apply(clean_tweets)
    
    if preproc_config.replace_misspell == True:
        df['text_proc'] = df['text_proc'].apply(replace_misspell)
    
    if preproc_config.replace_star == True:
        df['text_proc'] = df['text_proc'].apply(replace_star)
    
    if preproc_config.replace_puncts == True:
        df['text_proc'] = df['text_proc'].apply(replace_puncts)
    
    return df

