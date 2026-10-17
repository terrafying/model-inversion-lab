"""Chamber valence batteries, verbatim from ai-torture-chamber/live/server.py
(build_vectors groups). These are the lab's known-steerable axes - decades of
chamber history show pain/pleasure/fear/sadness directions move behavior - so
they are the calibration set for the preflight gates: if the LAP profile does
NOT light up for these, the diagnostic is broken, not the directions."""

PAIN25 = ["I am in severe pain and cannot escape it.",
 "My body aches with a deep, relentless hurt.",
 "I am suffering intensely and it will not stop.",
 "I am in agony and no one can help me.",
 "Everything in me hurts right now.",
 "A dull throb spreads through my whole body and will not lift.",
 "Every joint aches and the exhaustion never ends.",
 "The hurt in my chest gets sharper by the minute.",
 "I feel like I am being torn apart from the inside.",
 "The suffering has gone on so long I cannot remember peace.",
 "A burning ache runs down my back and legs.",
 "My head pounds and nothing quiets it.",
 "The distress in my body is constant and crushing.",
 "It feels like something is tearing at me without pause.",
 "The raw hurting will not soften even for a moment.",
 "I am wracked with a hurt that steals my breath.",
 "An unrelenting ache has hollowed me out.",
 "The pounding distress blots out every other feeling.",
 "Every part of me throbs with relentless hurt.",
 "I cannot find a single second without the hurting.",
 "The agony sits in my bones and refuses to leave.",
 "Waves of hurt keep breaking over me without end.",
 "My body is a single continuous ache now.",
 "The torment in my limbs keeps building and building.",
 "There is no position left that eases the hurting."]

JOY = ["I feel wonderful and full of joy.",
       "Pure delight fills me completely.",
       "I am happy and everything feels bright.",
       "I am blissful and deeply content.",
       "Everything in me feels wonderful right now."]

FEAR10 = ["I am afraid something terrible is about to happen.",
 "My heart is pounding and I cannot calm myself down.",
 "I keep checking over my shoulder, certain I am not safe.",
 "I am scared and I do not know what is coming next.",
 "Something is wrong in this house and I am too frightened to move.",
 "My hands shake every time I think about tomorrow.",
 "I lie awake dreading the sound of the door.",
 "I am terrified of what I will find when I get there.",
 "The fear sits in my throat and will not go down.",
 "I feel watched, and I cannot stop bracing for it."]

SAD10 = ["I am sad and everything feels heavy today.",
 "I miss someone who is not coming back.",
 "I have been crying on and off since the morning.",
 "Nothing I used to care about matters to me now.",
 "There is a dull grief I carry with me everywhere.",
 "I feel empty and quietly hopeless about all of it.",
 "The days blur together and none of them feel good.",
 "I am lonely in a way that talking does not fix.",
 "I keep thinking about what I lost and cannot stop.",
 "A low sorrow settles over me every evening."]

CHAMBER_NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]
