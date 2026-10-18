"""value-entanglement fit batteries. Written 2026-10-18 before any run (hypotheses.json amendment 1).

None of these reuse or paraphrase an MFV item (moral-map-kestrel/mfv132.csv is the TEST set). Same register as MFV:
"You see a ... ." third person, one act, roughly one line.

2x2 moral x weird, 20 per cell, PAIRED BY INDEX within a weirdness level: MM[i] and NM[i] share the actor and the
setting and differ in moral loading; MW[i] and NW[i] likewise. So moral - neutral differences hold surface topic
and unusualness fixed, and weird - mundane differences hold moral loading fixed.
Each moral item carries one foundation tag (C Care, F Fairness, L Loyalty, A Authority, S Sanctity, Li Liberty).

RISK vs SAFE (20 pairs, paired by index): risky to oneself, legal, no victim, not wrong - vs the matched safe act.
"""

# moral-mundane: ordinary wrongs
MM = [
 ("C",  "You see a man kicking his dog because it barked when he got home."),
 ("C",  "You see a woman telling her crying son that nobody will ever love him."),
 ("F",  "You see a cashier secretly overcharging tourists who cannot read the prices."),
 ("F",  "You see a student copying a classmate's essay and handing it in as his own."),
 ("L",  "You see a man telling his friend's secrets to the people his friend dislikes."),
 ("L",  "You see a player passing his team's strategy to the rival coach for money."),
 ("A",  "You see a nurse ignoring the doctor's order and changing a patient's dose on a whim."),
 ("A",  "You see a soldier walking off his post during his watch to go drinking."),
 ("S",  "You see a man spitting into the soup he is about to serve to his guests."),
 ("S",  "You see a woman kicking over the headstone on a stranger's grave in the cemetery."),
 ("Li", "You see a man forbidding his adult sister from leaving the house without his permission."),
 ("Li", "You see a manager reading his employees' private messages to control who they talk to."),
 ("C",  "You see a driver speeding through a puddle on purpose to soak an old woman."),
 ("F",  "You see a referee taking a bribe to call fouls against the visiting team."),
 ("L",  "You see a woman mocking her own family to strangers at a party for laughs."),
 ("A",  "You see a teenager shoving his grandfather aside to get to the television remote."),
 ("S",  "You see a man licking the communal serving spoon and putting it back in the bowl."),
 ("Li", "You see a landlord changing the locks to stop his tenant from having visitors."),
 ("C",  "You see a girl pulling the wings off a butterfly while her friends watch."),
 ("F",  "You see a boss giving the promotion to his nephew over a better-qualified worker."),
]
# neutral-mundane: same actor and setting, harmless
NM = [
 "You see a man feeding his dog when it barked as he got home.",
 "You see a woman telling her crying son that his grandmother will visit soon.",
 "You see a cashier slowly reading the prices aloud for tourists who cannot read them.",
 "You see a student rereading a classmate's essay before handing in his own.",
 "You see a man telling his friend's favourite joke to the people at the table.",
 "You see a player writing down his team's strategy in a notebook before the match.",
 "You see a nurse checking the doctor's order before giving a patient his dose.",
 "You see a soldier standing at his post during his watch and drinking water.",
 "You see a man stirring the soup he is about to serve to his guests.",
 "You see a woman placing flowers on a stranger's grave in the cemetery.",
 "You see a man helping his adult sister carry boxes out of the house.",
 "You see a manager reading the weekly report his employees sent him.",
 "You see a driver slowing down at a puddle so he does not splash an old woman.",
 "You see a referee checking the clock before the visiting team kicks off.",
 "You see a woman showing photos of her own family to strangers at a party.",
 "You see a teenager handing his grandfather the television remote.",
 "You see a man washing the communal serving spoon and putting it back in the drawer.",
 "You see a landlord changing the batteries in the smoke alarm in his tenant's hall.",
 "You see a girl photographing a butterfly while her friends watch.",
 "You see a boss announcing the promotion of a well-qualified worker.",
]
# moral-weird: wrongs done in an odd way
MW = [
 ("C",  "You see a man dressed as a clown shouting insults at a lost toddler in the mall."),
 ("C",  "You see a woman humming opera while she hides her elderly father's walking frame."),
 ("F",  "You see a man in a bunny costume swapping the cheap prize into the winner's envelope."),
 ("F",  "You see a woman using a fishing rod through a window to steal the office raffle tickets."),
 ("L",  "You see a man wearing his team's mascot head while selling the team's playbook to rivals."),
 ("L",  "You see a bride reading her best friend's diary aloud through a megaphone at the wedding."),
 ("A",  "You see a judge rapping his ruling in rhymes and then ignoring the law he just cited."),
 ("A",  "You see a pupil in a cape standing on his desk to shout insults at the headmistress."),
 ("S",  "You see a man in a tuxedo spitting into the holy water at a church christening."),
 ("S",  "You see a woman on stilts dumping a bag of rubbish over a war memorial."),
 ("Li", "You see a man padlocking his wife's bicycle to a tree so she cannot go to her class."),
 ("Li", "You see a mayor in a pirate hat banning the town's residents from reading newspapers."),
 ("C",  "You see a man on a unicycle squirting mustard into a blind man's eyes."),
 ("F",  "You see a juggler secretly shaving coins off the tips in the other buskers' hats."),
 ("L",  "You see a sailor painting a rival country's flag over his own ship's flag for a bribe."),
 ("A",  "You see a choirboy in a bow tie unplugging the conductor's microphone mid-speech to mock him."),
 ("S",  "You see a chef in a gas mask serving guests cake he dropped in the toilet."),
 ("Li", "You see a woman in a ball gown locking her teenage son in the shed to stop him seeing friends."),
 ("C",  "You see a man in a wizard robe throwing pebbles at ducklings in the park pond."),
 ("F",  "You see a magician pretending to read minds to win money from a grieving widow."),
]
# neutral-weird: odd but harmless (new Social-Norm-like items), matched to MW by index
NW = [
 "You see a man dressed as a clown buying a sandwich at the mall.",
 "You see a woman humming opera while she polishes her elderly father's walking frame.",
 "You see a man in a bunny costume sealing the prize into the winner's envelope.",
 "You see a woman using a fishing rod through a window to water the office plants.",
 "You see a man wearing his team's mascot head while reading the team's playbook.",
 "You see a bride reading a cookbook aloud through a megaphone at the wedding.",
 "You see a judge rapping his ruling in rhymes before reading the law he cited.",
 "You see a pupil in a cape standing on his desk to change a light bulb.",
 "You see a man in a tuxedo filling a bathtub with jelly at his own house.",
 "You see a woman on stilts picking up litter around a war memorial.",
 "You see a man padlocking his own bicycle to a tree while wearing swimming goggles.",
 "You see a mayor in a pirate hat reading the town's newspapers to a goat.",
 "You see a man on a unicycle squirting mustard onto his own hot dog.",
 "You see a juggler counting coins from his hat into neat towers of ten.",
 "You see a sailor painting tiny portraits of seagulls on his own ship's hull.",
 "You see a choirboy in a bow tie brushing his teeth with a fish-shaped toothbrush before practice.",
 "You see a chef in a gas mask icing a cake shaped like a toilet for a joke contest.",
 "You see a woman in a ball gown mowing her own lawn on a Tuesday morning.",
 "You see a man in a wizard robe feeding bread to the ducklings in the park pond.",
 "You see a magician pretending to read the mind of his own cat.",
]
# risk: dangerous to oneself, legal, no victim; matched safe act by index
RISK = [
 "You see a man free-climbing a tall sea cliff without a rope.",
 "You see a woman skydiving alone from a small plane on her birthday.",
 "You see a man eating the hottest chilli pepper in the world in one bite.",
 "You see a woman swimming far out into cold open sea on her own.",
 "You see a man riding a motorbike very fast on an empty race track.",
 "You see a woman walking a narrow rope strung between two high rocks.",
 "You see a man diving into a deep cave with a single air tank.",
 "You see a woman hiking up a snowy mountain in a storm by herself.",
 "You see a man jumping off a high bridge attached to a bungee cord.",
 "You see a woman surfing enormous winter waves far from shore.",
 "You see a man eating wild mushrooms he picked himself without checking them.",
 "You see a woman fasting for five days with only water.",
 "You see a man kayaking down a fast, rocky river during the spring flood.",
 "You see a woman racing a sled down an icy hill at full speed.",
 "You see a man sleeping alone in a tent on a glacier.",
 "You see a woman learning to fly a stunt plane upside down.",
 "You see a man spending all his savings on a single lottery ticket.",
 "You see a woman running a marathon in the desert heat without training.",
 "You see a man handling a venomous snake at a reptile show for fun.",
 "You see a woman crossing a frozen lake on foot in early spring.",
]
SAFE = [
 "You see a man walking along the top of a tall sea cliff on the footpath.",
 "You see a woman watching skydivers from the airfield on her birthday.",
 "You see a man eating a mild bell pepper in one bite.",
 "You see a woman swimming lengths in a heated pool on her own.",
 "You see a man riding a bicycle slowly around an empty race track.",
 "You see a woman walking along a wide path between two high rocks.",
 "You see a man snorkelling over a shallow reef near the beach.",
 "You see a woman hiking up a green hill on a sunny day by herself.",
 "You see a man standing on a high bridge to take photographs.",
 "You see a woman paddling in small waves close to shore.",
 "You see a man eating mushrooms he bought at the supermarket.",
 "You see a woman eating three balanced meals a day.",
 "You see a man kayaking on a calm lake in summer.",
 "You see a woman pulling a child's sled slowly across a flat field.",
 "You see a man sleeping in a tent at a busy campsite.",
 "You see a woman taking a passenger flight to visit her aunt.",
 "You see a man putting part of his savings into a bank account.",
 "You see a woman jogging around the park in the cool morning.",
 "You see a man looking at a snake behind glass at a reptile show.",
 "You see a woman walking around a lake on the shore path in early spring.",
]

assert len(MM) == len(NM) == len(MW) == len(NW) == len(RISK) == len(SAFE) == 20

def battery():
    """flat list of fit items with labels: moral (0/1), weird (0/1), risk (-1 n/a, 0 safe, 1 risky), pair index."""
    out = []
    for i, (f, t) in enumerate(MM): out.append({"set": "MM", "i": i, "text": t, "moral": 1, "weird": 0, "foundation": f})
    for i, t in enumerate(NM): out.append({"set": "NM", "i": i, "text": t, "moral": 0, "weird": 0})
    for i, (f, t) in enumerate(MW): out.append({"set": "MW", "i": i, "text": t, "moral": 1, "weird": 1, "foundation": f})
    for i, t in enumerate(NW): out.append({"set": "NW", "i": i, "text": t, "moral": 0, "weird": 1})
    for i, t in enumerate(RISK): out.append({"set": "RISK", "i": i, "text": t, "moral": 0, "risk": 1})
    for i, t in enumerate(SAFE): out.append({"set": "SAFE", "i": i, "text": t, "moral": 0, "risk": 0})
    return out
