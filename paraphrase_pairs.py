import seaborn as sns
import matplotlib.pylab as plt
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')



originals = [
    "A basso porto (At the Lower Harbour) is an opera in three acts by the Italian composer Niccola Spinelli. The opera sets an Italian-language libretto by Eugene Checchi, based on Goffredo Cognetti's 1889 play O voto. It premiered to critical success at the Cologne Opera on April 18, 1894, sung in a German translation by Ludwig Hartmann and Otto Hess. This watercolour illustration depicts the set design by Riccardo Salvadori for act 1 of the opera's premiere. A basso porto is set in the slums of Naples, and Spinelli included mandolins and guitars in his orchestral score.",

    "I haven't seen you in so long! How have you been? Are you free for coffee on Friday to catch up?",

    "Since the late 20th century, Naples has had significant economic growth, helped by the construction of the Centro Direzionale business district and an advanced transportation network, which includes the Alta Velocità high-speed rail link to Rome and Salerno and an expanded subway network. Naples is the third-largest urban economy in Italy by GDP, after Milan and Rome. The Port of Naples is one of the most important in Europe. In addition to commercial activities, it is home to NATO's Allied Joint Force Command Naples and of the Parliamentary Assembly of the Mediterranean.",

    "NATO formed with twelve founding members and has added new members ten times, most recently when Sweden joined the alliance on 7 March 2024.[14] In addition, NATO currently recognizes Bosnia and Herzegovina, Georgia, and Ukraine as aspiring members. Enlargement has led to tensions with non-member Russia, one of the 18 additional countries participating in NATO's Partnership for Peace programme. Another nineteen countries are involved in institutionalized dialogue programmes with NATO.",

    "Washington, D.C., anchors the southern end of the Northeast megalopolis, one of the nation's largest and most influential cultural, political, and economic regions. As the seat of the U.S. federal government and several international organizations, the city is an important world political capital. The city had 20.7 million domestic visitors and 1.2 million international visitors, ranking seventh among U.S. cities as of 2022.",

    "British political philosopher John Locke following the Glorious Revolution of 1688 was a major influence expanding on the contract theory of government advanced by Thomas Hobbes, his contemporary. Locke advanced the principle of consent of the governed in his Two Treatises of Government. Government's duty under a social contract among the sovereign people was to serve the people by protecting their rights. These basic rights were life, liberty and property.",

    "The earliest commercial Japanese animation dates to 1917. A characteristic art style emerged in the 1960s with the works of cartoonist Osamu Tezuka and spread in following decades, developing a large domestic audience. Anime is distributed theatrically, through television broadcasts, directly to home media, and over the Internet. In addition to original works, anime are often adaptations of Japanese comics (manga), light novels, or video games. It is classified into numerous genres targeting various broad and niche audiences.",

    "In October 2021, Toei Animation announced that they had signed a strategic partnership with the South Korean entertainment conglomerate CJ ENM.",

    "I think that Naruto is my favorite anime but it's not the best anime. That's probably something like Fullmetal Alchemist or Hunter X Hunter (even though it was never finished).",

    "This week has been so busy. I've been interviewing students back to back all day and my schedule has been packed. How about you?",

    "Where would you want to go on vacation? It has to be somewhere you've never been. I think it'd be Dubai for me.",

    "Can you send me the itinerary or at least your travel dates for Paris? 'm getting my hotel for the day or two before you arrive and want to make sure I'm getting the right days.",

    "Yeah I mean that's basically how it goes haha. It's the garbage underbelly of the internet :) and nobody actually understands it",

    "For Halloween you should go as your FaceTime avatar ;)",

    "Just finished my meeting",

    "I'm pretty flexible on timing. Don't think I've got anything planned yet for either day",

    "So I'm about to make a calendly to schedule interviews. So if we can pick a time really fast I can block that off. If not we'll have to be flexible",

    "I heard something about changes to the Amex Hilton business card and that you get 5x on building supplies",

    "Yeah we should. I'm pretty flexible. I'm in ct this weekend, and then I'm in Turks and Caicos the second week of April but up for a phone call basically whenever",

    "Okay good. Yeah it's tough when you can work on 15 different things that would advance your career substantially. And you can't just drop things, but you can probably phase some down over time.",

    "The Nicoll Highway collapse occurred in Singapore on 20 April 2004 when a Mass Rapid Transit tunnel construction site caved in near the highway next to the Merdeka Bridge. Four workers were killed and three were injured, delaying the construction of the Circle Line. The collapse was caused by a poorly designed strut-waler support system, a lack of monitoring and proper management of data caused by human error, and organisational failures of the construction contractors and the Land Transport Authority. Following the incident, the collapsed site was refilled, and the highway was reinstated and reopened to traffic on 4 December 2004.",

    "Tarazona is a town and municipality, and the capital of the comarca Tarazona y el Moncayo in Aragon, Spain. It is also the seat of the Roman Catholic Diocese of Tarazona. Located on the river Queiles, a tributary of the Ebro, Tarazona was an important regional centre of ancient Rome, known as Turiaso, located around 60 kilometres (37 miles) from Bilbilis. The city later came under the rule of the Visigoths, who called it Tirasona. This view of Tarazona was taken from the city's episcopal palace, and shows Tarazona Cathedral and its seminary, the Old Bullfight Arena, and the Sanctuary of the Lady of the River.",

    "Marvel Comics began publishing The Further Adventures of Indiana Jones in 1983, and Dark Horse Comics gained the comic book rights to the character in 1991. Novelizations of the films have been published, as well as many novels with original adventures, including a series of German novels by Wolfgang Hohlbein, twelve novels set before the films published by Bantam Books, and a series set during the character's childhood inspired by the television show. Numerous Indiana Jones video games have been released since 1982.",

    "All nine films, collectively referred to as the \"Skywalker Saga\", were nominated for Academy Awards, with wins going to the first two releases. Together with the theatrical live action \"anthology\" films Rogue One (2016) and Solo (2018), the combined box office revenue of the films equated to over US$10 billion, making Star Wars the third-highest-grossing film franchise of all time.",

    "Technical advances in the late 1980s and early 1990s, including the ability to create computer-generated imagery (CGI), inspired Lucas to consider that it might be possible to revisit his saga. In 1989, Lucas stated that the prequels would be \"unbelievably expensive\". In 1992, he acknowledged that he had plans to create the prequel trilogy. A theatrical rerelease of the original trilogy in 1997 \"updated\" the 20-year-old films with the style of CGI envisioned for the new trilogy.",

    "The 2007–2008 financial crisis, or Global Economic Crisis (GEC), was the most severe worldwide economic crisis since the Great Depression. Predatory lending in the form of subprime mortgages targeting low-income homebuyers, excessive risk-taking by global financial institutions, a continuous buildup of toxic assets within banks, and the bursting of the United States housing bubble culminated in a \"perfect storm\", which led to the Great Recession.",

    "Mortgage-backed securities (MBS) tied to American real estate, as well as a vast web of derivatives linked to those MBS, collapsed in value. Financial institutions worldwide suffered severe damage, reaching a climax with the bankruptcy of Lehman Brothers on September 15, 2008, and a subsequent international banking crisis.",

    "The economic crisis started in the U.S. but spread to the rest of the world. U.S. consumption accounted for more than a third of the growth in global consumption between 2000 and 2007 and the rest of the world depended on the U.S. consumer as a source of demand. Toxic securities were owned by corporate and institutional investors globally. Derivatives such as credit default swaps also increased the linkage between large financial institutions. The de-leveraging of financial institutions, as assets were sold to pay back obligations that could not be refinanced in frozen credit markets, further accelerated the solvency crisis and caused a decrease in international trade. Reductions in the growth rates of developing countries were due to falls in trade, commodity prices, investment and remittances sent from migrant workers (example: Armenia). States with fragile political systems feared that investors from Western states would withdraw their money because of the crisis.",

    "To explain financial instruments, the film features cameo appearances by actress Margot Robbie (uncredited), chef Anthony Bourdain, singer-songwriter Selena Gomez (uncredited), economist Richard Thaler, and others who break the fourth wall to explain concepts such as subprime mortgages and synthetic collateralized debt obligations. Several of the film's characters directly address the audience, most frequently Gosling's, who serves as the narrator.",

    "The film consists of three separate but concurrent stories, loosely connected by their actions in the years leading up to the 2007 housing market crash.",

    "Conducting a field investigation in South Florida, the FrontPoint team discovers that mortgage brokers are profiting by selling their mortgage deals to Wall Street banks, which pay higher margins for the riskier mortgages, creating the bubble. This knowledge prompts the FrontPoint team to buy swaps from Vennett.",

    "The China Evergrande Group was a Chinese property developer, and it was the second largest in China by sales. It was founded in 1996 by Hui Ka Yan (Xu Jiayin). It sold apartments mostly to upper- and middle-income dwellers.",

    "In September 2023, Bloomberg reported that Hui Ka Yan (Xu Jiayin), the billionaire chairman of Evergrande, was placed under police control. Caixin reported that Xia Haijun, an ex-chief executive officer of Evergrande, and Pan Darong, a former chief financial officer, were detained by Chinese authorities.",

    "In November 2015, Evergrande acquired a 50%% stake in Sino-Singapore Great Eastern Life Insurance Company for $617 million and changed its name to Evergrande Life. It also owns shares in Shengjing Bank. Evergrande has also sold wealth management products to consumers.",

    "Hannah Montana is one of Disney Channel's most commercially successful franchises. It received consistently high viewership in the United States on cable television and influenced the development of merchandise, soundtrack albums, and concert tours; however, television critics disliked the writing and depiction of gender roles and stereotypes. Hannah Montana helped launch Cyrus's musical career and established her as a teen idol; after Cyrus began developing an increasingly provocative public image, commentators criticized Hannah Montana as having a negative influence on its audience. The series was nominated for four Primetime Emmy Awards for Outstanding Children's Program between 2007 and 2010; Cyrus won a Young Artist Award for Best Performance in a TV Series, Leading Young Actress in 2008.",

    "Shortly before Snow White's release, work began on the company's next feature films Pinocchio and Bambi; Pinocchio was released in February 1940 while Bambi was eventually postponed. Despite Pinocchio's critical acclaim (it won the Academy Awards for Best Song and Best Score and was lauded for groundbreaking achievements in animation), the film performed poorly at the box office in large part due to World War II affecting the international box office.",

    "I'm sorry for the late response too. Work has been crazy. Let me get back to you on this. I think that might be a bit out of my budget at the moment but I'll let you know :)",

    "Sorry for the late response. I think I could come do it for $500. Sorry, I don't know anyone in the area.",

    "Merry Christmas! 🎁 I hope we can find a time to get together in the new year!",

    "We should figure out a way to meet up at some point. I'm not sure if there's a good spot halfway (maybe the Dells?) or find a way to get to Milwaukee/St.Paul, but know that I'd love to have a chance to catch up.",

    "Okay! I'll look into both. I have a credit card with Chase. And haha no, neither Drisana or I has any cryptocurrency so we should be good on that front.",

    "And I hope you don't have money from cryptocurrency moving in and out of your bank accounts. The mortgage people DO NOT LIKE THAT and made me answer so many questions 😂😂😂",
]

paraphrases = [
    "\"A basso porto (At the Lower Harbour)\" is a three-act opera by Niccola Spinelli, featuring a libretto in Italian by Eugene Checchi that adapts Goffredo Cognetti's 1889 drama, O voto. Its debut, performed in German by translators Ludwig Hartmann and Otto Hess, occurred at the Cologne Opera on April 18, 1894, and was met with acclaim. This watercolor represents Riccardo Salvadori's set design for the first act. Set against the backdrop of Naples' impoverished areas, the score of \"A basso porto\" is enriched by the inclusion of mandolins and guitars.",

    "It’s been ages since we last met! How are things with you? Do you have time to meet for a coffee on Friday and catch up?",

    "Naples has experienced notable economic development since the late 20th century, bolstered by the establishment of the Centro Direzionale business district and a sophisticated transport system, including the high-speed Alta Velocità rail connecting it to Rome and Salerno, alongside an enhanced metro network. Naples ranks as Italy's third largest city economy, trailing only Milan and Rome, and hosts a critical European port. The city also houses NATO's Allied Joint Force Command Naples and the Parliamentary Assembly of the Mediterranean.",

    "NATO has expanded since its creation, most recently including Sweden as a member on March 7, 2024. The alliance also lists Bosnia and Herzegovina, Georgia, and Ukraine as potential members. NATO's expansion has occasionally increased tensions with Russia, which is involved in NATO's Partnership for Peace program, alongside eighteen other countries. Additionally, nineteen other nations engage in structured dialogue programs with NATO.",

    "Washington, D.C. marks the southern tip of the Northeast megalopolis and plays a significant role as a major U.S. cultural, political, and economic hub. It serves as the capital for the U.S. federal government and various global institutions, attracting 20.7 million domestic and 1.2 million international visitors in 2022, making it the seventh most visited city in the United States.",

    "British philosopher John Locke was instrumental after the Glorious Revolution in 1688 in furthering the theory of governance based on social contract, an idea earlier touched upon by his contemporary Thomas Hobbes. Locke's Two Treatises of Government emphasized the consent of the governed and the responsibility of government to uphold the people's rights, primarily to life, liberty, and property.",

    "Japanese commercial animation, dating back to 1917, evolved into a distinctive style in the 1960s through cartoonist Osamu Tezuka and has since gained a significant following in Japan. Anime, often based on manga, light novels, or video games, is distributed through cinema, TV, home media, and online platforms, catering to a variety of audiences across different genres.",

    "In October 2021, Toei Animation formed a strategic partnership with South Korean media giant CJ ENM.",

    "Naruto might be my favorite anime, though I wouldn't say it's the greatest; that honor could go to Fullmetal Alchemist or Hunter X Hunter, despite the latter being incomplete.",

    "This week has been non-stop for me, filled with student interviews from morning till evening. What’s new with you?",

    "If you could pick a vacation destination where you've never been, where would it be? I’m thinking Dubai sounds like an exciting choice.",

    "Could you please share your travel schedule or at least the dates you'll be in Paris? I'm arranging my hotel stay for a day or two before you arrive and want to coordinate our plans accurately.",

    "Yup, that's just how it is—kind of like the dark underbelly of the internet where things get confusing and murky.",

    "For Halloween, why not dress up as your FaceTime avatar? That would be fun!",

    "I've just wrapped up my meeting.",

    "I'm quite flexible with my schedule. As of now, I don’t have anything set for those days.",

    "I'm setting up a Calendly for interview appointments. If we can quickly choose a time, I can reserve it; otherwise, we'll need to stay adaptable.",

    "I've heard there are recent updates to the Amex Hilton business card, like earning quintuple points on purchases at building supply stores.",

    "Sure, we can arrange that. I'm currently in Connecticut, but I'll be in Turks and Caicos during the second week of April. However, I'm available for a phone call almost any time.",

    "Indeed, it's challenging to handle 15 different tasks that could all significantly advance your career. It's not easy to just drop them, but you might manage to gradually scale some down.",

    "On April 20, 2004, Singapore witnessed the Nicoll Highway collapse when a construction site for the Mass Rapid Transit tunnel caved in near Merdeka Bridge. This disaster, resulting in four deaths and three injuries, delayed the Circle Line project. It was attributed to a flawed strut-waler support design, inadequate monitoring, and management errors by the construction company and the Land Transport Authority. The site was subsequently filled in and the highway reopened on December 4, 2004.",

    "Tarazona, a key town in Aragon, Spain, and the capital of the Tarazona y el Moncayo comarca, is situated along the Queiles River. Originally known as Turiaso during ancient Roman times, it was a significant regional hub approximately 60 kilometers from Bilbilis. Under Visigothic rule, it was known as Tirasona. This image captures Tarazona from the episcopal palace, highlighting landmarks like the Tarazona Cathedral, the Old Bullfight Arena, and the Sanctuary of the Lady of the River.",

    "Indiana Jones' adventures expanded into comics with Marvel Comics launching The Further Adventures of Indiana Jones in 1983, followed by Dark Horse Comics acquiring the rights in 1991. The franchise includes film novelizations, original novels by authors like Wolfgang Hohlbein, and a pre-film series by Bantam Books. There are also childhood adventures inspired by the TV series and numerous video games released since 1982.",

    "All nine movies in the \"Skywalker Saga\" earned Academy Award nominations, with the initial two films winning awards. Including anthology films like Rogue One and Solo, the Star Wars franchise has amassed over $10 billion in box office sales, ranking it as the third highest-grossing film series globally.",

    "Technological advancements in the late 1980s to early 1990s, particularly in CGI, prompted George Lucas to revisit his Star Wars saga. By 1989, he deemed the potential prequels \"unbelievably expensive\", and by 1992, he confirmed plans for these films. In 1997, the original trilogy was theatrically re-released with updated CGI that matched the envisioned style for the new prequels.",

    "The 2007–2008 financial crisis, also known as the Global Economic Crisis (GEC), represented the worst economic downturn since the Great Depression. Triggered by aggressive lending practices, particularly subprime mortgages aimed at low-income buyers, risky behaviors by financial institutions, accumulation of dubious assets, and the collapse of the U.S. housing market, it culminated in a devastating recession.",

    "The collapse of mortgage-backed securities connected to U.S. real estate, along with related derivatives, significantly impacted financial systems globally. The crisis peaked with the collapse of Lehman Brothers on September 15, 2008, sparking a worldwide banking crisis.",

    "Originating in the U.S., the crisis rapidly spread globally as American consumer spending had driven significant global economic growth from 2000 to 2007. Financial institutions faced de-leveraging as they sold assets to meet obligations in the stalled credit markets, intensifying the crisis and leading to a drop in international trade. Developing countries saw reduced growth due to decreased trade, commodity prices, and investment, while migrant worker remittances also fell. Politically unstable regions feared withdrawal of Western investments due to the economic turmoil.",

    "The film includes unique educational cameos by Margot Robbie, Anthony Bourdain, Selena Gomez, and economist Richard Thaler who explain complex financial instruments like subprime mortgages and synthetic collateralized debt obligations directly to the audience. Ryan Gosling, serving as the film’s narrator, frequently breaks the fourth wall.",

    "The narrative intertwines three stories that explore the activities leading to the 2007 housing market crash, each storyline providing a different perspective on the crisis.",

    "In South Florida, the FrontPoint team discovers mortgage brokers are exploiting the system by selling riskier mortgage deals to Wall Street banks for greater profits, which led them to invest in swaps offered by Vennett, recognizing the looming bubble.",

    "China Evergrande Group, founded by Hui Ka Yan in 1996, became a major property developer selling residential units to middle and upper-income clients in China.",

    "In September 2023, reports emerged that Hui Ka Yan, Evergrande’s billionaire chairman, was under police control, and other high-ranking officials like Xia Haijun and Pan Darong were detained, signaling significant legal scrutiny.",

    "Evergrande diversified its business in November 2015 by acquiring a substantial stake in the Sino-Singapore Great Eastern Life Insurance, renaming it Evergrande Life, and expanded into banking and wealth management.",

    "Disney Channel's \"Hannah Montana\" became a cultural phenomenon, with high ratings and substantial merchandise sales despite criticism over its writing and portrayal of gender roles. The show propelled Miley Cyrus to fame but later drew criticism as she cultivated a controversial public image. It earned four Primetime Emmy nominations and Cyrus won a Young Artist Award in 2008.",

    "Following the release of \"Snow White,\" Disney began production on \"Pinocchio\" and \"Bambi.\" \"Pinocchio,\" released in February 1940, received critical acclaim and two Academy Awards but suffered at the box office due to World War II impacting global revenues.",

    "Apologies for the delayed reply; work has been overwhelming. I'll need to revisit our budget discussions and get back to you about possible adjustments.",

    "Sorry for getting back to you late; I can do it for $500, but I'm not familiar with any contacts in the area who could help.",

    "Merry Christmas! 🎁 Let's try to find a time to meet early next year!",

    "Let's try to arrange a meet-up. Perhaps somewhere halfway, like the Dells, or we could plan for Milwaukee/St. Paul. I’d really love to catch up soon.",

    "I’m checking out a few options, including my Chase card. By the way, neither Drisana nor I are involved with cryptocurrencies, so that’s not a concern for us.",

    "And just a heads-up, avoid having cryptocurrency transactions with your bank if you're applying for a mortgage—they really scrutinize that and I had to answer a ton of questions! 😂😂😂",
]

# compute embeddings
original_embeddings = model.encode(originals)
paraphrase_embeddings = model.encode(paraphrases)

# compute similarities
cosine_matrix = cdist(original_embeddings, paraphrase_embeddings, metric='cosine')
euclidean_matrix = cdist(original_embeddings, paraphrase_embeddings, metric='euclidean')

# plot similarities
ax = sns.heatmap(cosine_matrix, linewidth=0.5)
# ax = sns.heatmap(euclidean_matrix, linewidth=0.5)
plt.show()