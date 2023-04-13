import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = "A neutron star is the collapsed core of a massive supergiant star, which had a total mass of between 10 and 25 solar masses, possibly more if the star was especially metal-rich.[1] Neutron stars are the smallest and densest stellar objects, excluding black holes and hypothetical white holes, quark stars, and strange stars.[2] Neutron stars have a radius on the order of 10 kilometres (6.2 mi) and a mass of about 1.4 solar masses.[3] They result from the supernova explosion of a massive star, combined with gravitational collapse, that compresses the core past white dwarf star density to that of atomic nuclei."

prompt = '''
0:00
it all started as I was hunting for a
0:02
chat gpt-like model that was open
0:04
sourced of good quality and locally
0:06
runnable on a single consumer GPU so
0:09
ideally under 24 gigabytes of memory I
0:13
came across things like GPT for all in
0:15
together computers 20 Bill and seven
0:17
Bill chat models based on a Luther ai's
0:20
Neo X models and others like pythia but
0:23
then I stumbled upon chat glm 6B a
0:29
relatively tiny model capable of running
0:31
on as little as six gigabytes of memory
0:34
at int4 with quantization or 13
0:37
gigabytes at half Precision for
0:39
inference not only is this model
0:42
extremely small it's quite effective at
0:44
just single query and response as well
0:47
as multiple back and forths with some
0:49
decently sized contacts it's definitely
0:52
not a full chat GPT but it's shockingly
0:56
good for its size for example we can
0:58
just ask it to do a typical
1:00
summarization task and then we can ask
1:02
it to instead condense that down to a
1:04
single sentence this not only
1:06
exemplifies a bit of context from chat
1:09
history and using it and being smart in
1:11
sort of like this kind of back and forth
1:13
chat context but it's also got a pretty
1:15
good understanding of the important
1:17
elements both in the longer
1:18
summarization giving us the bits that
1:20
actually mattered as well as being able
1:22
to drop the entire article down to a
1:24
single sentence pretty well at this
1:27
point I think most of you guys are
1:28
pretty familiar with the things that
1:31
large language models can do so I'm just
1:33
not going to go through everything in
1:34
the video just showing you a bunch of
1:36
examples that you've probably seen many
1:37
times with many different models but
1:39
there are quite quite a few pitfalls
1:41
that smaller models usually have first a
1:46
long enough chat context will begin to
1:49
confuse the model and smaller models
1:51
tend to just plain make mistakes way
1:54
more often way more quickly or even
1:55
almost always if you don't fine tune
1:57
them or restrict them in some way with
1:59
like pre-prompting and or like even
2:02
fine-tuning via training but chat glm 6B
2:05
was surprisingly good comparatively to
2:08
other models of similar size and even a
2:11
little bigger maybe to 10 or 20 billion
2:13
even and it was good like right out of
2:16
the box so I dug a tiny bit because up
2:19
to this point I had never even
2:20
personally heard of this glm doing a
2:24
Google search or even a chat GPT search
2:26
it tells me that glm stands for
2:29
generalized linear model and I was
2:31
pretty confident that this was not what
2:34
the glm stood for in this specific
2:36
context so first I found the original
2:38
glm paper from the 17th of March 2022
2:42
which defines glm as a general language
2:45
model now from my very brief
2:48
understanding and first read through it
2:50
does seem as though a glm is still a
2:52
generative pre-trained Transformer model
2:54
but the glm does seem to be
2:57
differentiated by bi-directional
2:59
attention as compared to unidirectional
3:01
attention that most other GPT models use
3:03
as well as the use of the gaussian error
3:08
linear unit
3:09
instead of the traditional rectified
3:11
linear unit or relio the difference of
3:14
these activation functions is extremely
3:16
obvious upon graphing the gel U gel you
3:20
I'm not really sure I'm going to I guess
3:21
I'll say gel you is a smoother function
3:24
with a non-zero derivative for all the
3:26
inputs it's slightly more complicated to
3:29
actually calculate as shown here in the
3:31
code which also makes it a bit more
3:34
computationally intense so to speak but
3:36
the authors of the glm paper argue that
3:38
it is worth the trade-off and that the
3:40
trade-off is essentially negligible when
3:42
scaled out to these large language
3:44
models anyways in general some research
3:48
has also suggested that the gel U is
3:50
also it's just more useful for very
3:53
large and deep neural networks which
3:55
suffer from issues like Vanishing
3:57
gradients so from here the glm-130b
4:01
paper or 130 billion parameters paper
4:04
can be found from the 5th of October
4:06
2022. I'll link both of these papers in
4:08
the description by the way if I forget
4:09
someone just remind me and I'll put them
4:10
in now this is the first open source
4:12
large-scale implementation of a glm that
4:16
I could find and again it does seem as
4:18
though even this glm is very structured
4:20
very similar to gpt3 and that the main
4:23
differentiator here is going to be that
4:25
gel U activation function in the
4:26
bi-directional attention so this is this
4:30
glm 130 billion parameter model is meant
4:33
to be comparable to gpd3 which has 175
4:37
billion parameters but the goal was to
4:39
make this comparable model capable of
4:42
running on much more consumer like
4:44
Hardware so in this case glm-130b
4:47
through infor with quantization can run
4:49
on as minimal as four 30 90s to put this
4:53
in perspective gpt3 for inference is
4:55
going to want something like a 150 000
4:58
machine or more whereas glm-130b is
5:02
going to want something more like a 30
5:04
to 40 000 machine now I know you if you
5:07
go into like the used Market or you
5:08
might be able to get deals in other ways
5:10
and finagle even less money but at the
5:12
end of the day it's like a fifth of the
5:13
cost so um this is a pretty substantial
5:16
difference on what will be the End
5:19
Hardware to run inference with these
5:21
models what will be the requirement and
5:23
this is just a staggering difference
5:25
we can also compare glm-130b to models
5:28
like opt and Bloom and at least given
5:30
the benchmarks that the authors have
5:31
provided it does appear that glm-130b is
5:35
extremely competitive at generative
5:37
tasks natural language understanding
5:38
tasks and multilingual tasks noting that
5:42
glm-130b is a mixed model of both
5:44
English and Chinese although I would say
5:47
it seems to be more favoring towards
5:50
Chinese but it's a little unclear to me
5:53
the data that's used at least in glm
5:55
130b will get to glm chat glm 6B in a
5:58
moment again I will link both the
5:59
original glm paper as well as the
6:01
glm-130b paper as both of these actually
6:04
have some pretty interesting insights
6:06
but the glm-130b paper also goes into
6:08
some more depth and detail about
6:11
training very large models especially
6:14
when compared to more medium-sized
6:16
models like 10 billion parameters or
6:17
less versus you know what does it take
6:19
to do a 100 billion parameter model also
6:22
one of the other insights that they
6:23
brought up which I have kind of out this
6:25
way for a while I haven't seen it in the
6:27
research but apparently this is a thing
6:28
that there seems to be a large amount of
6:32
people that seem to think that these
6:34
very very large models are all almost
6:36
across the board suffering from not
6:38
being fully trained in my opinion the
6:41
first model that I really felt this way
6:42
on was and is the bloom 176 billion
6:46
parameter model if you look at the
6:48
tensorboard graph it just is it's very
6:50
obvious when you look at it that
6:52
training did not finish like they just
6:54
didn't finish training it and like all
6:57
these other models including the 130
6:58
Bill model
7:00
especially if you want to open source
7:02
these models or share them with the
7:03
world you need grants you need somebody
7:05
to sponsor this and you usually only get
7:07
a certain fixed period of time to get it
7:09
done and given all the other hardships
7:12
that you're going to come across as you
7:13
try to do this it just happens to be the
7:16
case more often than not that these
7:17
models just probably aren't actually
7:19
even being fully trained now this all
7:21
brings us to chat glm 6B which is 6
7:24
billion parameter which as of yet no
7:27
paper is offered that I can find yet but
7:30
there is a blog post with some useful
7:32
ish information once you convert it from
7:34
Chinese to English but I do hope they
7:36
release a paper eventually here too on
7:39
what did they do did they do anything
7:40
really different RL rlhf wise so
7:45
reinforcement learning through human
7:46
feedback wise but mainly though chat glm
7:49
has 6.2 billion parameters a context
7:51
length in training of 2048 and is
7:55
intended to run on as minimal as a
7:57
single 2080 TI I can also see that
8:00
there's a chat glm main model so a
8:04
full-sized chat glm with 100 billion
8:07
parameters but this one doesn't appear
8:09
to be open sourced yet and is being
8:11
slowly rolled out to researchers and
8:13
it's a little unclear to me if they ever
8:16
intend to release the full-size chat glm
8:18
but we'll see so I cannot test this but
8:21
the authors note that chat glm is most
8:24
successful with with Chinese dialogue
8:26
obviously I can't actually test this and
8:28
I believe this is true with chat glm the
8:30
full size and Chad glm 6B I can only
8:33
test on English
8:34
and this isn't really super shocking
8:37
given that the research is mainly from a
8:38
group at a Chinese University that said
8:41
getting back to where all this began for
8:42
me personally is I find that the chat
8:45
glm 6B model is just the best for this
8:50
kind of like use 10-ish billion
8:53
parameter model size but even compared
8:56
to some of the 20 billion parameter
8:58
models I find chat glm 6B to be just
9:02
generally better now I think both all of
9:05
this requires much more time to kind of
9:07
dive in and kind of poke around and see
9:09
or maybe figure out what's the best way
9:11
to sort of prompt some of these smaller
9:13
models it might just be the case that
9:15
the way I'm used to prompting models
9:16
just happens to work better with chat
9:18
glm6b but I just can't help but feel
9:20
like man this is a really cool model
9:22
it's really fast and it's really small
9:24
that memory footprint it is exceptional
9:27
and then from there it is a bit unclear
9:29
to me at this point if this sort of all
9:32
of these successes has more to do with
9:34
say the glm attributes of the
9:38
bi-directional attention or the gel U
9:39
activation function or possibly is it
9:42
the multi-task training that they did so
9:44
again uh at least with the glm models
9:46
for the 130b billion parameter model
9:48
they did things like text generation but
9:51
then also specific tasks uh I think it
9:55
was like five percent or something was
9:56
like more specific masking tasks so it's
9:59
curious if maybe that has to has some
10:03
sort of roll into it uh also this is one
10:06
of the I think few ish models where you
10:09
have a very uh very large body of text
10:13
of both Chinese and English and we do
10:16
have lots of multilingual models but
10:18
those multilingual models tend to suffer
10:20
pretty substantially on something like
10:23
Chinese because Chinese is such a vastly
10:26
different language than something like
10:28
English and the difference between
10:29
Chinese and English is going to be just
10:31
huge compared to maybe English and I
10:34
don't know Spanish or something like
10:35
that the differences are so staggering
10:37
that you might even say that this is
10:39
more of a multi-modal model or for sure
10:43
like a multi-task type model and we've
10:45
already seen that these doing that sort
10:47
of thing to these models you would have
10:49
thought that oh for any specific task it
10:51
would be better to just make it you know
10:53
train a model to do that specific task
10:54
and you'll have the most success that
10:55
way but that is proving to be very false
10:58
especially as we make models bigger and
11:00
bigger it does seem like multitask
11:02
multimodal all this stuff seems to
11:05
benefit even those singular tasks that
11:07
they might do so getting back to how I
11:09
got here in the first place to use this
11:11
model it's exceptionally simple to
11:12
actually get going I'm going to link the
11:14
hugging face space for The Unofficial
11:16
demo in the description so you can
11:18
really just you can just go there and
11:19
play with it in your browser see if it
11:21
works for maybe some of your ideas but
11:23
you can also copy that code and run that
11:25
demo completely locally download the
11:27
actual model the way it's all that you
11:28
could fine tune it from here if you
11:30
wanted you can also just run it
11:32
headlessly you don't have to run it
11:33
through this radio app or you don't have
11:35
to make a chat app out of it you can
11:37
make it do other things maybe you want
11:38
to make an app that summarizes text for
11:40
example so you could really just do
11:43
whatever the heck you want with it so
11:45
anyways that was a pretty interesting
11:47
Journey going through all this stuff in
11:49
the glm and I'm curious to see if glm is
11:52
going to just simply Encompass the
11:54
Chinese variants of these essentially
11:56
GPT style models or will we start seeing
12:00
much more of this bi-directional
12:01
attention going on as well as the Glu
12:04
activation function you know who knows
12:06
like the gpt4 could have been using gel
12:09
you we just we don't know anything
12:10
because open AI is no not being open
12:12
about uh gpt4 the other thing that I
12:16
really appreciate here is it is clear
12:18
that through all of these models the
12:21
researchers aren't just thinking like
12:23
okay let's make a bigger model or okay
12:25
let's make a model that does this and
12:26
let's just you know pick some arbitrary
12:28
size
12:29
they went into this thinking about okay
12:32
well chat glm 6B for example we want to
12:35
be able to fit this on a 2080 TI I think
12:38
it was a 2080 TI or a 2080 but whatever
12:39
the case 6 billion or six gigs of memory
12:42
same thing with uh glm 130b we want to
12:46
be able to run this on consumer Hardware
12:48
so for on a 4 30 90 setup
12:52
um if you could call that consumer but
12:54
it's at least work you know consumer
12:56
workstation let's say you didn't need to
12:58
buy uh v100s a100s RTX 8000s you didn't
13:02
have to buy any of these like server
13:03
grade cards and pay that server grade
13:05
price and that is the difference between
13:09
um uh that that is that like 5x markup
13:12
like the server grade cards are just so
13:15
much more expensive I at least
13:16
personally hope that as time goes on
13:18
researchers more and more continue
13:20
thinking about things in terms of okay
13:22
well what would be the hardware
13:23
requirements of this who could run it
13:25
what size should we go with you know
13:27
start thinking in that way more often
13:29
because sometimes times it does seem
13:30
like these models are just arbitrarily
13:32
sized I also really appreciate the
13:34
breakdown of the major issues that the
13:36
researchers encountered in the training
13:38
of
13:39
glm-130b in this like technical report
13:42
section of their paper they basically go
13:44
through and kind of like outline some of
13:45
the major uh issues like unforeseen
13:48
issues that they came across as they
13:50
tried to train one of these very very
13:51
large models anyways that's chat glm 6B
13:54
definitely a cool model to check out
13:56
again links to everything will be in the
13:57
description and even if you don't have a
14:00
GPU to run this on you can still play
14:01
with it in your browser and it's
14:02
definitely worth checking out this like
14:04
arguably very tiny model that is really
14:07
good so all that said there are still
14:10
quite a few models coming soon and
14:13
there's probably going to be a lot of
14:14
competition in the space going forward
14:16
and even some of the existing models uh
14:19
like from a Luther Ai and to you know
14:22
together computer and some of these
14:23
others are still in like active training
14:26
and they're getting like updates and
14:28
stuff so there's definitely going to be
14:29
a lot of compensation in the space going
14:31
forward and I'm I'm super excited to
14:33
kind of see what other people come up
14:35
with as time goes on I'd also like to
14:37
take another pass at even like the the
14:40
pythia for example a variance from a
14:42
Luther Ai and the chat variants from
14:44
together computer which those seem to be
14:47
very intertwined I need to like do a
14:49
little more research into those and
14:50
figure out like who played what role but
14:53
it seems like those are all based on
14:54
these like Neo X or gptj and those kinds
14:58
of models so it's like it seems like a
14:59
Luther Ai and together computer are kind
15:00
of teaming up to make these chat uh
15:03
variants but at least again
15:04
in my short experience and experiments
15:08
um I would say that chat glm 6B is the
15:11
best small chat style GPT chat GPT like
15:15
model that you can run locally on as
15:17
little as six uh six gigabytes of memory
15:21
so that's all for now if you're looking
15:23
to learn more about neural networks and
15:24
how they work you can check out the
15:25
neural networks from scratchbook at
15:26
nnfs.io otherwise I will see you all in
15:30
another video and also if you have a
15:33
chat uh things that you want me to look
15:34
at if you think I didn't do something
15:36
Justice or you disagree or whatever feel
15:39
free to comment below I'm curious to see
15:40
what what you guys are thinking so
15:42
there's a lot of these things I look
15:43
into and like people are saying one
15:46
thing and then as soon as I voice my
15:47
opinion like I'm like I actually don't
15:49
really like this or I think this one's
15:50
really cool and I find out like the
15:52
public sentiment or just like the
15:53
sentiment people have does seem to vary
15:55
greatly and um it's interesting to hear
15:58
what other people are thinking and
15:59
feeling and finding and all that so
16:00
anyways that's all for now I will see
16:02
you all in another video
'''

SAMPLE_RESPONSE_IF_MAX_RETURN_TOKENS_IS_60 = '''
: Chat GLM 6B is an open-source chat GPT-like model that can run on a single consumer GPU with under 24GB of memory and can be used for summarization and back-and-forth conversations. It is fast, efficient, and has a good memory footprint.
'''

SAMPLE_RESPONSE_IF_MAX_RETURN_TOKENS_IS_2000 = '''
: Chat GLM-6B is an open source model with 10 billion parameters, capable of running on consumer grade hardware and good for summarization tasks. It works with English and Chinese, but causes some unexpected issues while training. Chat GLM-130B is a larger model with 130 billion parameters for more complex tasks.
'''

SAMPLE_RESPONSE_IF_MAX_RETURN_TOKENS_IS_2000_2 = '''
: Chat GLM 6B is a 6.2 billion parameter GPT model designed for Chinese dialogue that runs on consumer hardware and is considered one of the best small chat-style GPT models. It uses multi-task training, bi-directional attention, and potentialy Glu activation functions. There are more models coming soon, with opinions varying greatly on which is best.
'''

DEFAULT_PROMPT = "Tl;dr"
DEFAULT_MODEL = "text-davinci-003"

MAX_TOKEN_LENGTH = 4097


def summarize_text(text, prompt=DEFAULT_PROMPT, model=DEFAULT_MODEL):
  try:
    if model == "gpt-3.5-turbo":
      messages = [{"role": "user", "content": f"{text}\n\n{prompt}"}]
      response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
      )
      # print(response)
      # global r
      # r = response
      return response["choices"][0]["message"]["content"]
    else:
      response = openai.Completion.create(model=model,
                                          prompt=f"{text}\n\n{prompt}",
                                          temperature=0.7,
                                          max_tokens=2000,
                                          top_p=1.0,
                                          frequency_penalty=0.0,
                                          presence_penalty=1)
      return response["choices"][0]["text"]
  except openai.error.InvalidRequestError as error:
    if error.user_message.startswith("This model's maximum context length is"):
      text_part_1 = text[:int(len(text) / 2)]
      text_part_2 = text[int(len(text) / 2 + 1):]
      summary_part_1 = summarize_text(text_part_1, model=model)
      summary_part_2 = summarize_text(text_part_2, model=model)
      return summarize_text(summary_part_1 + summary_part_2, model=model)


print(summarize_text(prompt, model="gpt-3.5-turbo"))
