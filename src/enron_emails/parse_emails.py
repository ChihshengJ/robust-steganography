import json
import os
import quopri
import re
import unicodedata
from operator import itemgetter

import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)  # Download required NLTK data


def parse_enron_emails(directory_path):
    """
    Reads all files in the given directory, parses emails on the delimiter (two blank lines),
    and returns a list of email bodies.

    Parameters:
        directory_path (str): Path to the directory containing EnronSent files.

    Returns:
        list: A list of email bodies as strings.
    """
    email_list = []

    for filename in os.listdir(directory_path):
        if filename.startswith("enronsent"):
            file_path = os.path.join(directory_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                # Read the file content and split on two blank lines
                content = file.read()
                emails = content.split("\n\n")  # Split by the two blank lines

                # Decode each email and clean up the text
                for email in emails:
                    if email.strip():
                        decoded_email = quopri.decodestring(email).decode(
                            "utf-8", errors="ignore"
                        )
                        cleaned_email = decoded_email.replace(
                            "- 30 -", ""
                        ).strip()  # Remove '- 30 -'
                        email_list.append(cleaned_email)

    return email_list


def clean_text(text):
    """
    Cleans text by removing common artifacts and unwanted patterns.
    """
    # Clean up control characters while preserving normal text
    cleaned_chars = []
    for char in text:
        # Get unicode category for this character
        category = unicodedata.category(char)

        # Keep normal letters, numbers, punctuation, spaces, and symbols
        # Remove control characters (Cc) and format characters (Cf)
        if category not in ["Cc", "Cf"]:
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")

    text = "".join(cleaned_chars)

    # Skip if text contains Enron-specific patterns
    if re.search(r"E\d{3}[-\s]?\d*", text):  # Matches E### patterns
        return ""

    # Skip if too many formatting characters
    if sum(c in "|>+-=" for c in text) > len(text) * 0.1:  # More than 10% special chars
        return ""

    # Remove ASCII art and formatting characters
    text = re.sub(r"\|[\-\+\>\s]+\|", " ", text)  # Vertical bars with formatting
    text = re.sub(r"[\-\+\>\<\|]{4,}", " ", text)  # Repeated formatting chars

    # Remove specific patterns
    text = re.sub(
        r"\b(?:SBX2|SB|AB)\s*\d+\b", "[BILL]", text
    )  # Legislative bill numbers

    # Common boilerplate patterns to remove
    artifacts = [
        "=====================",
        "-----Original Message-----",
        "- 30 -",
        "____________________",
        "-----",
        "All rights reserved",
        "This message was sent from",
        "If you have any questions or comments regarding this email",
        "Copyright",
        "©",
        "Confidential",
        "CONFIDENTIAL",
        "Best regards",
        "Kind regards",
        "Sincerely",
        "Tel:",
        "Phone:",
        "Fax:",
        "Email:",
        "www.",
    ]

    # Skip paragraphs that are likely signatures or contact info
    if any(
        line.strip().startswith(("Tel:", "Phone:", "Fax:", "Email:", "www."))
        for line in text.split("\n")
    ):
        return ""

    cleaned = text.strip()

    # Remove common artifacts
    for artifact in artifacts:
        cleaned = cleaned.replace(artifact, "")

    # Remove lines that are just URLs, email addresses, or other noise
    lines = []
    for line in cleaned.split("\n"):
        line = line.strip()
        # Skip if line is too short or just punctuation
        if (
            not line
            or len(line) < 3
            or line in ["?", ".", "!"]
            or
            # Skip email headers
            re.match(r"\d{2}/\d{2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(?:AM|PM)", line)
            or re.match(r"(?:From|To|Subject|Date|Sent):\s*", line)
            or
            # Skip URLs and emails
            any(
                pattern in line.lower()
                for pattern in ["http://", "https://", "@", ".com", ".net", ".org"]
            )
            or
            # Skip if too many quote markers
            line.count(">") > 5
            or
            # Skip if looks like a file path
            line.count(".") > 3
        ):
            continue
        lines.append(line)

    cleaned = " ".join(lines)

    # Skip if text appears truncated
    if cleaned.endswith((":", ",", "...")):
        return ""

    # Skip if too short after cleaning
    if len(cleaned) < 10:
        return ""

    return cleaned


def parse_paragraphs(emails):
    """
    Splits a list of emails into individual paragraphs, keeping only those with 2+ sentences.
    """
    seen_paragraphs = set()  # Track duplicates
    paragraphs = []

    for email in emails:
        email_paragraphs = [p.strip() for p in email.split("\n\n")]

        for paragraph in email_paragraphs:
            clean_para = clean_text(paragraph)

            if not clean_para:  # Skip empty paragraphs
                continue

            # Skip if we've seen this paragraph before
            if clean_para in seen_paragraphs:
                continue

            # Only keep paragraphs with 2+ valid sentences
            sentences = sent_tokenize(clean_para)
            valid_sentences = [
                s
                for s in sentences
                if len(s.strip()) > 3  # Increased minimum length
                and not s.strip().startswith(">")
                and not any(
                    s.strip().startswith(p)
                    for p in [
                        "http",
                        "@",
                        "===",
                        "tel:",
                        "fax:",
                        "phone:",
                        "email:",
                        "www.",
                    ]
                )
                and not any(
                    boilerplate in s.lower()
                    for boilerplate in [
                        "all rights reserved",
                        "confidential",
                        "copyright",
                    ]
                )
            ]

            if len(valid_sentences) >= 2:
                seen_paragraphs.add(clean_para)
                paragraphs.append(clean_para)

    return paragraphs


def plot_sentence_distribution(paragraphs):
    """
    Creates a histogram showing the distribution of sentences per paragraph.

    Parameters:
        paragraphs (list): List of paragraph strings
    """
    # Calculate sentence counts
    sentence_counts = [len(sent_tokenize(p)) for p in paragraphs]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(sentence_counts, bins=range(0, 11, 1), edgecolor="black", alpha=0.7)
    plt.title("Distribution of Sentences per Paragraph")
    plt.xlabel("Number of Sentences")
    plt.ylabel("Number of Paragraphs")
    plt.xlim(0, 10)
    plt.grid(True, alpha=0.3)

    # Add text box with statistics
    stats_text = f"Total Paragraphs: {len(paragraphs)}\n"
    stats_text += f"Mean Sentences: {sum(sentence_counts) / len(sentence_counts):.1f}\n"
    stats_text += f"Max Sentences: {max(sentence_counts)}"
    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.show()


def save_sorted_paragraphs(paragraphs, output_file):
    """
    Sorts paragraphs by sentence count and saves them to a JSON file with metadata.

    Parameters:
        paragraphs (list): List of paragraph strings
        output_file (str): Path to output JSON file
    """
    # Create list of paragraphs with metadata
    paragraph_data = []
    for p in paragraphs:
        sentences = sent_tokenize(p)
        paragraph_data.append(
            {
                "text": p,
                "sentence_count": len(sentences),
                "sentences": sentences,
                "word_count": len(p.split()),
            }
        )

    # Sort by sentence count
    sorted_paragraphs = sorted(paragraph_data, key=itemgetter("sentence_count"))

    # Save to JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "total_paragraphs": len(sorted_paragraphs),
                    "mean_sentences": sum(
                        p["sentence_count"] for p in sorted_paragraphs
                    )
                    / len(sorted_paragraphs),
                    "max_sentences": max(
                        p["sentence_count"] for p in sorted_paragraphs
                    ),
                    "total_words": sum(p["word_count"] for p in sorted_paragraphs),
                },
                "paragraphs": sorted_paragraphs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


# Process the emails
directory_path = os.path.expanduser("~/Downloads/enronsent")
emails = parse_enron_emails(directory_path)
multi_sentence_paragraphs = parse_paragraphs(emails)

# Save sorted paragraphs
output_file = "enron_paragraphs.json"
save_sorted_paragraphs(multi_sentence_paragraphs, output_file)

# Print basic statistics and plot histogram
print(f"Total multi-sentence paragraphs found: {len(multi_sentence_paragraphs)}")
print(f"Saved to: {output_file}")
print("\nFirst 5 multi-sentence paragraphs:")
for i, paragraph in enumerate(multi_sentence_paragraphs[:5], 1):
    print(f"\nParagraph {i}:\n{paragraph}")
    print(f"Number of sentences: {len(sent_tokenize(paragraph))}")

plot_sentence_distribution(multi_sentence_paragraphs)
