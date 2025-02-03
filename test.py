import json

from ai_memory.cognitive.memory import SemanticMemory

chat_history = [
    {"role": "user", "content": "Can you explain Einstein's theory of relativity in simple terms?"},
    {"role": "assistant",
     "content": "Of course! Einstein's theory of relativity consists of two main parts: Special Relativity and General Relativity. Special Relativity states that the speed of light is always constant, no matter how fast you're moving, leading to time dilation and length contraction. General Relativity extends this idea to gravity, describing it as the warping of spacetime by massive objects."},

    {"role": "user", "content": "That sounds fascinating! What do you mean by 'warping of spacetime'?"},
    {"role": "assistant",
     "content": "Great question! Imagine spacetime as a flexible sheet, like a trampoline. If you place a heavy object, like a bowling ball, in the middle, it creates a dent. Smaller objects, like marbles, will roll toward it, not because they're pulled, but because the surface is curved. This is how massive objects like planets and stars bend spacetime, creating what we perceive as gravity."},

    {"role": "user", "content": "So is that why time runs slower near a black hole?"},
    {"role": "assistant",
     "content": "Exactly! The stronger the gravitational field, the more spacetime bends, and this includes time itself. Near a black hole, gravity is so strong that time slows down significantly compared to someone far away. This effect is called gravitational time dilation and has been confirmed through experiments, like observing time differences in clocks placed at different altitudes on Earth."},

    {"role": "user",
     "content": "Does that mean if I traveled near a black hole and came back, I would age slower than people on Earth?"},
    {"role": "assistant",
     "content": "Yes, that's a real possibility! If you orbited close to a black hole for a while and then returned to Earth, you would have aged less than those who stayed here. This idea was beautifully illustrated in the movie *Interstellar*, where one hour on a planet near a black hole equaled seven years on Earth."},

    {"role": "user", "content": "That’s mind-blowing! Does this mean time travel to the future is possible?"},
    {"role": "assistant",
     "content": "In a way, yes! Since time slows down in strong gravitational fields or at high speeds (as predicted by relativity), traveling near light speed or staying near a black hole could allow you to move into the future relative to others. However, traveling back in time is a whole other challenge, requiring exotic physics that we haven't proven yet!"}
]

# Print the chat history for reference
pair = []
pair_list = []
memory = SemanticMemory("./test_semantic_memory")

for message in chat_history:
    pair.append(f"<{message['role'].capitalize()}> {message['content']} </{message['role'].capitalize()}>\n")
    if len(pair) == 2:
        out = ''.join(pair)
        pair_list.append(out)
        pair = []
        memory.store(out)

print(json.dumps(pair_list, indent=4))
chat_history_2 = [
    {"role": "user",
     "content": "I've been thinking more about our discussion on relativity. You mentioned time dilation near a black hole—does that mean astronauts could use this to extend their lifespan?"},
    {"role": "assistant",
     "content": "Great follow-up! Yes, in theory, an astronaut orbiting near a black hole, in a region with extreme gravitational time dilation, would experience time much slower than people on Earth. This means they would effectively age slower, relative to those who remained on Earth. However, surviving near a black hole would be incredibly challenging due to extreme tidal forces and radiation."},

    {"role": "user",
     "content": "Right, that makes sense. You also mentioned that moving at high speeds can slow time. Would it be possible to build a spaceship that travels near the speed of light and allows astronauts to time travel into the future?"},
    {"role": "assistant",
     "content": "Yes, that's a real consequence of Special Relativity! If a spaceship traveled at 99% the speed of light, time for those on board would pass much slower compared to people on Earth. So, an astronaut on a high-speed journey could return to find that centuries have passed on Earth, even though they only experienced a few years."},

    {"role": "user",
     "content": "That sounds like something out of a sci-fi movie! But wouldn't accelerating to such speeds require a huge amount of energy?"},
    {"role": "assistant",
     "content": "Absolutely! According to relativity, as an object approaches the speed of light, its mass effectively increases, meaning it requires exponentially more energy to accelerate further. This makes it practically impossible to reach light speed with conventional technology. Scientists are exploring concepts like nuclear propulsion and antimatter engines, but for now, we're far from achieving such speeds."},

    {"role": "user",
     "content": "You also mentioned in our last discussion that traveling back in time would require 'exotic physics.' What kind of theories explore that?"},
    {"role": "assistant",
     "content": "Good memory! Some theoretical models suggest that time travel to the past might be possible under special conditions. One idea involves 'closed timelike curves,' predicted in solutions to Einstein's equations, like those involving rotating black holes (Kerr black holes) or wormholes. However, these require exotic matter with negative energy, which hasn’t been observed in nature."},

    {"role": "user",
     "content": "So if wormholes were real, could they connect different points in time as well as space?"},
    {"role": "assistant",
     "content": "Exactly! A traversable wormhole, if it existed, could theoretically link two distant locations in space and even different times. But stabilizing a wormhole would require negative energy, and even if one existed naturally, we have no idea how to control or navigate it. For now, it's more of a theoretical possibility than a practical one."},

    {"role": "user",
     "content": "Fascinating! I love how relativity keeps challenging our intuition about time and space."},
    {"role": "assistant",
     "content": "It really does! Einstein revolutionized our understanding of reality, showing that time isn’t absolute and space can bend. The universe is far stranger than we ever imagined!"}
]


# Print the second chat history for reference
for message in chat_history_2:
    if message["role"] == "user":
        out = f"{message['role'].capitalize()}: {message['content']}"
        results = memory.recall(out, 3)
        print(out)
        for r in results:
            print(f"\nMemory: {r['content']}")
            print(f"Type: {r['memory_type']}")
            print(f"Similarity: {r['similarity']:.3f}")