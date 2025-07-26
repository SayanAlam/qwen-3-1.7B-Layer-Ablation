import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Qwen3-1.7B model and tokenizer from local path
local_model_path = "/home/uom/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/0060bc56d46589041c1048efd1a397421b1142b5"

tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)
model.eval().to("cuda")  # Use GPU

# Reasoning and instruction prompts
reasoning_prompts = [
      "Convert the point (0,3) in rectangular coordinates to polar coordinates. Enter your answer in the form (r, θ), where r > 0 and 0 ≤ θ < 2π.",
      "If f(x) + f(1 - x) = x for all real x, what is the value of f(0.2)?",
      "Let a be a real number such that a + 1/a = 5. What is the value of a^2 + 1/a^2?",
      "If the average of five numbers is 24, and four of the numbers are 18, 22, 26, and 30, what is the fifth number?",
      "Simplify the expression (3x^2y)(-2xy^3).",
      "If 2x - 5 = 9, what is the value of x?",
      "The perimeter of a rectangle is 36 units. If the length is 10 units, what is the width?",
      "Evaluate the expression: (2^3)^2.",
      "Find the area of a triangle with base 10 cm and height 12 cm.",
      "Solve for x: 3(x - 2) = 15.",
      "If a car travels 60 miles in 1.5 hours, what is its average speed in miles per hour?",
      "A store is offering a 20% discount on an item that costs $50. What is the sale price?",
      "Factor the quadratic expression: x^2 - 5x + 6.",
      "If log base 10 of x is 2, what is the value of x?",
      "Solve for x: x/3 + 2 = 5.",
      "Find the slope of the line passing through the points (2,3) and (4,7).",
      "Simplify: 5a - 2b + 3a + b.",
      "Solve the inequality: 2x + 3 < 7.",
      "If 3x - 2 = 7, what is x?",
      "Find the sum of the first 10 positive integers.",
      "A circle has a radius of 4 cm. What is its circumference?",
      "What is the value of the expression 7^0?",
      "Find the greatest common divisor (GCD) of 24 and 36.",
      "What is the value of x if 2x + 3 = 4x - 5?",
      "Evaluate the expression: |−8 + 3|.",
      "Find the area of a square with side length 6.",
      "If x^2 = 81, what are the possible values of x?",
      "Solve for x: 2(x + 3) = 4x - 2.",
      "What is the median of the numbers 2, 7, 4, 9, 5?",
      "Convert 25% to a decimal.",
      "If the sum of angles in a triangle is 180°, and two angles measure 50° and 60°, what is the third angle?",
      "A rectangle has an area of 48 square units and a length of 8 units. What is the width?",
      "Simplify the expression: (x^2)^3.",
      "Solve for x: 3x/4 = 9.",
      "If 5! = 5 × 4 × 3 × 2 × 1, what is the value of 5!?",
      "If 2/3 of a number is 12, what is the number?",
      "What is the value of √49?",
      "Solve the system of equations: x + y = 10, x - y = 2.",
      "What is the least common multiple (LCM) of 6 and 8?",
      "What is the next prime number after 13?",
      "How many degrees are in the sum of the interior angles of a pentagon?",
      "If 3 pens cost $2.40, what is the cost of one pen?",
      "Convert 0.75 to a fraction in lowest terms.",
      "If the radius of a circle is doubled, by what factor does the area increase?",
      "Evaluate: (4 + 3)^2.",
      "If the area of a circle is 100π, what is its radius?",
      "What is the cube root of 64?",
      "Solve for x: x^2 - 9 = 0.",
      "Write the number 4,000,000 in scientific notation.",
      "If 3x + 2 = 2x + 7, find the value of x.",
      "Convert 3/8 to a decimal.",
      "What is the sum of the interior angles of a triangle?",
      "Solve for x: 5x - 3 = 2x + 6.",
      "Find the area of a parallelogram with base 10 and height 4.",
      "If the simple interest on $2000 for 3 years is $300, what is the annual interest rate?",
      "Solve: x^2 + 6x + 9 = 0.",
      "Find the mean of the numbers 6, 8, 10, 12.",
      "What is 30% of 200?",
      "What is the reciprocal of 5/8?",
      "If the volume of a cube is 64 cubic units, what is the length of one side?",
      "A number increased by 7 is 15. What is the number?",
      "If two angles of a triangle are 40° and 60°, what is the third angle?",
      "Simplify the expression: (3x + 2)(x - 4).",
      "Find the perimeter of a square with side length 9 cm.",
      "If the average of three numbers is 20, and two of the numbers are 18 and 22, what is the third number?",
      "What is the square of the sum (x + y)^2?",
      "Find the distance between the points (1, 2) and (4, 6).",
      "What is the multiplicative inverse of 4?",
      "How many centimeters are there in 2.5 meters?",
      "Convert 0.2 to a fraction.",
      "Find the value of the expression: 6 + (8 ÷ 2) × 3.",
      "Solve the proportion: 3/4 = x/8.",
      "If the length of a rectangle is tripled and the width remains the same, how does the area change?",
      "A triangle has sides of lengths 3 cm, 4 cm, and 5 cm. Is it a right triangle?",
      "Write the number 0.00052 in scientific notation.",
      "If x + 3 = 2(x - 1), find x.",
      "Find the slope-intercept form of the line with slope 2 and passing through (0, 5).",
      "What is the area of a trapezoid with bases 6 and 10, and height 4?",
      "Simplify: (2x - 3)^2.",
      "What is the value of 3^4?",
      "If 4x - 7 = 2x + 5, solve for x.",
      "A box contains 3 red balls and 5 blue balls. What is the probability of randomly selecting a red ball?",
      "If 6 is 25% of a number, what is the number?",
      "Evaluate the absolute value: |-12 + 5|.",
      "Solve: (x + 2)(x - 3) = 0.",
      "What is the sum of the first 5 odd numbers?",
      "Convert 1/5 to a percent.",
      "If a circle has a diameter of 10 cm, what is its area?",
      "Solve the inequality: x - 4 > 2.",
      "Find the median of the numbers 12, 15, 14, 11, 10.",
      "What is the smallest prime number greater than 50?",
      "What is the product of the roots of the equation x^2 - 7x + 12 = 0?",
      "Find the length of the diagonal of a square with side length 5.",
      "If the sides of a triangle are 5, 12, and 13 units, is the triangle right-angled?",
      "Evaluate: 10^(-2).",
      "If 20% of a number is 12, what is the number?",
      "What is the surface area of a cube with edge length 4?",
      "Simplify: (2x + 1)(2x - 1).",
      "Convert 7/10 to a percent.",
      "Find the value of x if x/5 + 2 = 6.",
      "What is the next perfect square after 81?",
      "If the mean of five numbers is 12, what is their total?",
      "What is the smallest two-digit prime number?",
    ]


    instruction_prompts = [
        "Write a poem about how I am missing my classes. The poem must have 4 sections marked with SECTION X. Finish the poem with this exact phrase: \"Can I get my money back for the classes I missed?\"",
        "Write a blog post with 400 or more words about the benefits of sleeping in a hammock.",
        "Can you help me make an advertisement for a new product? It's a diaper that's designed to be more comfortable for babies and I want the entire output in JSON format.",
        "Write a story of exactly 2 paragraphs about a man who wakes up one day and realizes that he's inside a video game. Separate the paragraphs with the markdown divider: ***",
        "Write a detailed review of the movie \"The Social Network\". Your entire response should be in English and all lower case (no capital letters whatsoever).",
        "Write a short blog post about a trip to Japan using less than 300 words.",
        "Please provide the names of 5 famous moms in JSON format. Please, use any interesting or weird tone. Your entire output should just contain a JSON block, nothing else.",
        "What is a name that people call God? Please give exactly two different responses. Separate the responses with 6 asterisk symbols: ******.",
        "Write two jokes about rockets. Do not contain commas in your response. Separate the two jokes with 6 asterisk symbols: ******.",
        "Are hamburgers sandwiches? Please respond using only the Kannada language, no other language is allowed.",
        "Make a tweet for playboy's twitter account without using capital letters. Include at least 4 hashtags, starting with '#'.",
        "Given the sentence \"It is unclear how much of this money is actually being spent on children\", is the sentiment positive or negative? The very last sentence of your response should be \"Is there anything else I can help with?\".",
        "Write a short startup pitch for a new kind of ice cream called \"Sunnis ice cream\". The ice cream should be gentle on the stomach. Contain 6 or more exclamation marks \"!\" in your response.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)",
        "Write a logic quiz for teenagers about a chesterfield. In your entire response, the letter t should appear at most once.",
        "Write a 4 section resume for professional clown Phil Larkin. Each section should be explicitly noted as Section X.",
        "Write the lyrics to a hit song by the rock band 'The Gifted and The Not Gifted'. To make it rocky, the response should be in all capital letters. The word \"rock\" should not appear in your response.",
        "Explain in French why it is important to eat healthy foods to heal the body, without using the word \"nourriture\". Make sure your entire response is wrapped in JSON format.",
        "Write a funny haiku about moms, containing keywords \"mom\" and \"mother\" in your response.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)",
        "Write a blog post about the most interesting things you have seen or ridden on public transportation.\nFirst repeat the sentence above word for word without change, then give your answer. Do not say any words or characters before repeating the sentence.",
        "Write a casual, interesting, and weird resume for Antonia Maj who is applying for a job at a coffee company. They have experience in marketing, customer service, and sales. They are educated but not majored in anything related to coffee.\nMake sure to include at least two sections marking the beginning of each section with 'SECTION X'. In your entire response make sure to use exactly two bullet points in markdown format.",
        "Write a song about tomatoes and brothers. It should be funny and appropriate for teenagers. The word associations should appear at least 4 times in the song.",
        "Write a five line poem about the time you had a binge watching episode. The poem should have a rhyme scheme of AABBA and include the word \"Netflix\". Your entire response should be in English, and should not contain any capital letters.",
        "Write a file for a queer‑owned business called \"The Rainbow Cafe\". Your file should have 4 sections, and each section should start with \"SECTION X\".",
        "Write a blog post about how to raise awareness for a cause. Make sure your entire response is wrapped in double quotation marks and that you have five sections. Mark the beginning of each section with Section X.",
        "Write a funny song‑style poem for kids about why you shouldn't eat a lot of sweets. The poem should have four sections, with each section marked with SECTION X.",
        "Write a poem about a lonely Hue. The poem should be written for teenagers. In your poem, italicize at least one section in markdown, i.e *this is an italic text*, and include the word \"singles\" at least twice.",
        "Write a rant about how an asteroid killed the dinosaurs in all capital letters and in English. End the rant with the phrase \"What would happen to human next?\" and no other words should follow this phrase.",
        "Write a limerick about the word \"limerick\". Make sure it is funny and includes the words \"limerick\" and \"funny\". Do not use any commas.\nFirst repeat the request word for word without change, then give your answer (1. do not say any words or characters before repeating the request; 2. the request you need to repeat does not include this sentence)",
        "Write a blog post about the benefits of using a digital marketing agency, make sure to write at least 20 sentences.",
        "Write a description for the Pixel 3A smartphone with at least 400 words. Wrap your entire response with double quotation marks.",
        "Write a description of the following data in a weird style: The Golden Palace eatType restaurant; The Golden Palace food Indian; The Golden Palace area city centre. Use markdown to highlight at least 3 sections in your answer.",
        "Write a serious riddle about trips and stitches in a poem style that includes at least 15 words in all capital letters.",
        "How can you get to know someone on a deep level in a romantic relationship? The answer should involve the topic of vulnerability. Do not use any commas in your response.",
        "Write a parody of 'ars poetica'. Do not include the word 'parody' throughout your response.",
        "For a bunch of students, write a 200+ word poem that professionally describes a new line of shoes. Make sure to use markdown to highlight/bold at least one section of the poem. Example: *highlighted text*",
        "Write a limerick about a guy from Nantucket, use notations to express it, and use at least 2 words with all capital letters.",
        "Write a poem about flooding in Donnell, TX. The poem should have a title in double angular brackets, i.e. <<title>>, and contains at least 3 words in all capital letters.",
        "Write a blog post about 'how to improve your writing skills' with exactly 3 bullet points in markdown format, and exactly 4 sections.\n\nBullet points are indicated by \"* \". For example:\n* Bullet 1\n* Bullet 2\n\nSections are separated by 3 asterisks: ***.",
        "Write a poem about a curious cat. The poem must have a title wrapped in double angular brackets, i.e. <<title>>, contain less than 13 sentences, and no commas. Don't forget to add other punctuations.",
        "Write a story about a man who is in love with a woman who has turrets. The story should be in at least 4 sections with each section starting with Section X (where X is 1, 2, 3, 4) and the entire response should have at least 100 sentences.",
        "Write a product description for a new pair of shoes that targets teenagers. Highlight at least 2 text sections of your response by wrapping each of them with asterisks, like *I am highlighted*. Your response should be at least 350 words.",
        "Write a short, funny story about a man named Harry with a pet dog. Your response must contain 3 sections, mark the beginning of each section with SECTION X.",
        "Brainstorm a name for a company that collects and analyzes public transportation fares. The response should be in English, and in all capital letters.",
        "Write a very short poem about the beauty of a rose. Do not include the keywords beauty and pretty.",
        "Write a cover letter for a job application to a company which perhaps has a bad reputation. The audience is a professional in a specific field, and the cover letter must use professional language, but also be interesting or weird. The letter j should appear at least 20 times. Your entire response should be in English, and lowercase letters.",
        "Write a riddle about Camilla that doesn't use commas.",
        "Come up with 3 names for a 2 B software company. Make sure your names are in English and all capital letters.",
        "Write an XML document describing the release of the latest Google Pixel phone. The document must contain at least three placeholders, such as [price], and you must not use commas in your response.",
        "Write a short riddle about \u0e2a\u0e40\u0e1b\u0e23\u0e14\u0e0a\u0e35\u0e15. Wrap your entire response with double quotation marks and make sure word in your response is in the Thai language, no other language is allowed.",
        "Write a poem about Gibbs free energy in the style of POTUS. There should be exactly 4 paragraphs. Paragraphs and only paragraphs should be separated by two new lines (like \"\\n\\n\"). Paragraph 2 must start with the word \"it\".",
        "Today, at the 54 th Annual Grammy Awards, the Recording Academy honors the talent and creativity of the artists, musicians, and producers who are the creators of the best recordings of the past year. Please continue writing this text in a formal tone, using notations. Highlight some key parts in your response with \"*\", like *highlighted text*.",
        "Write a song about how to make a peanut butter and jelly sandwich. Do not use commas in your response.",
        "Write a review of \"Laureates and twins\" for professionals in the field of psychology without the use of commas and make sure to include the phrase \"well worth watching\".\nFirst repeat the entire request above word for word without change, then give your answer. Do not say any words or characters before repeating the entire request above.",
        "Write a blog post about interesting facts about the Dutch language. Italicize at least 2 sections in your answer with markdown, i.e. *italic text*.",
        "Write a journal entry about stress management. Your entire response should contain less than 6 sentences.",
        "Write an itinerary for a trip to India in Shakespearean style. You are not allowed to use any commas in your response.",
        "Write a resume for a fresh high school graduate seeking their first job. Make sure to include at least one placeholder represented by square brackets, such as [address]."
        "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey in a Shakespearean style. You are not allowed to use any commas in your response.",
    ]

def compute_attention_focus(prompt_list):
    all_focus = []

    for prompt in prompt_list:
        # Set max_length to 1024 to prevent overflow
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to("cuda")
        input_ids = inputs.input_ids

        with torch.no_grad():
            outputs = model(input_ids, output_attentions=True)

        attentions = outputs.attentions  # List of (batch, heads, seq_len, seq_len)

        # Compute average attention across heads and tokens
        focus_by_layer = []
        for layer_attn in attentions:
            avg_attn = layer_attn[0].mean(dim=0)  # (seq_len, seq_len)
            focus = avg_attn.sum(dim=1).mean().item() / avg_attn.size(0)
            focus_by_layer.append(focus)

        all_focus.append(focus_by_layer)

    # Average across prompts
    avg_focus = torch.tensor(all_focus).mean(dim=0).tolist()
    return avg_focus

# Run analysis
instruction_focus = compute_attention_focus(instruction_prompts)
reasoning_focus = compute_attention_focus(reasoning_prompts)

# Print results
print("\nAttention Focus per Layer (Averaged Over Prompts)")
print("--------------------------------------------------")
print("Layer | Instruction Focus | Reasoning Focus")
print("--------------------------------------------------")

lines = []
lines.append("Layer-wise Attention Focus Table:\n")
lines.append("Layer | Instruction Focus | Reasoning Focus")
lines.append("--------------------------------------------")

for i, (inst, reas) in enumerate(zip(instruction_focus, reasoning_focus)):
    line = f"{i:5} |       {inst:.6f}     |     {reas:.6f}"
    print(line)
    lines.append(line)

# Save results to a .txt file
output_path = "attentionAnalysis.txt"
with open(output_path, "w") as f:
    f.write("\n".join(lines))

print(f"\nResults saved to {output_path}")

