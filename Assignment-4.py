import numpy as np
import scipy as sp # may be useful to compute probabilities
import time # may be useful to check the execution time of some function
import math

"""
Please refer to lecture slides.
Please refer to README file.
All the functions that you define must be able to handle corner cases/exceptions
"""

"""
Starting and ending locations (indices) of red and green exons in the reference sequence - Begins

1. Red Exon Locations
"""
RedExonPos = np.array([
    [149249757, 149249868], # R1
    [149256127, 149256423], # R2
    [149258412, 149258580], # R3
    [149260048, 149260213], # R4
    [149261768, 149262007], # R5
    [149264290, 149264400]  # R6
    ])
"""
2. Green Exon Locations
"""
GreenExonPos = np.array([
    [149288166, 149288277], # G1
    [149293258, 149293554], # G2
    [149295542, 149295710], # G3
    [149297178, 149297343], # G4
    [149298898, 149299137], # G5
    [149301420, 149301530]  # G6
    ])
"""
Starting and ending locations (indices) of red and green exons in the reference sequence - Ends
"""    

INDICES = None

def loadLastCol(filename):
    """
    Input: Path of the file corresponding the last column (BWT).
    Output: The last column (BWT) in string format.
    """
    # function body - Begins

    with open(filename, "r") as f:
        LastCol = f.readlines()
    # Actually returns list of lines

    # function body - Ends
    return LastCol #string data type

def loadRefSeq(filename):
    """
    Input: Path of the file containing the reference sequence.
    Output: The reference sequence in string format.
    """
    # function body - Begins

    with open(filename, "r") as f:
        f.readline() # Discard the first line (header)
        RefSeq = f.readlines()
    # Actually returns list of lines

    # function body - Ends
    return RefSeq # string data type

def loadReads(filename):
    """
    Input: Path of the file containing all the reads.
    Output: A list containing all the reads.
    """
    # function body - Begins

    with open(filename, "r") as f:
        Reads = f.readlines()
    # Actually returns list of lines

    # function body - Ends
    return Reads # list of strings

def loadMapToRefSeq(filename):
    """
    Input: Path of the file containing mapping of the first column to the reference sequence.
    Output: numpy integer array containing the mapping.
    """
    # function body - Begins

    with open(filename, "r") as f:
        MapToRefSeq = f.readlines()
    # Actually returns list of lines

    # function body - Ends
    return MapToRefSeq # numpy integer array

def getIndices(bwt):
    """
    Input: The bwt last column
    Output: Dictionary of delta milestones (number of "A", "C", etc. at every 100-interval position)
    """
    rows = len(bwt)
    # Map of milestones for the characters "A", "C", "G" and "T"
    indices = { "A": np.zeros(rows, dtype=int),
                "C": np.zeros(rows, dtype=int),
                "G": np.zeros(rows, dtype=int),
                "T": np.zeros(rows, dtype=int) }
    
    # Computer milestones for first row, which is independent
    indices["A"][0] = bwt[0].count("A")
    indices["C"][0] = bwt[0].count("C")
    indices["G"][0] = bwt[0].count("G")
    indices["T"][0] = bwt[0].count("T")

    # Compute milestones for subsequent rows, which depends on
    # the previous row because it is cumulative
    for i in range(1, rows):
        indices["A"][i] = bwt[i].count("A") + indices["A"][i - 1]
        indices["C"][i] = bwt[i].count("C") + indices["C"][i - 1]
        indices["G"][i] = bwt[i].count("G") + indices["G"][i - 1]
        indices["T"][i] = bwt[i].count("T") + indices["T"][i - 1]
    
    return indices

def getBand(read, bwt, indices):
    # Initial band is the whole string
    band_0 = 0
    band_1 = indices["A"][-1] + indices["C"][-1] + indices["G"][-1] + indices["T"][-1] + 1

    for i, chr in enumerate(reversed(read)):
        band_0_loc = band_0 // 100
        band_0_off = band_0 % 100
        band_1_loc = band_1 // 100
        band_1_off = band_1 % 100

        # Rank of the character to be searched
        band_0_t = indices[chr][band_0_loc] - bwt[band_0_loc][band_0_off:].count(chr) + 1
        band_1_t = indices[chr][band_1_loc] - bwt[band_1_loc].count(chr) + bwt[band_1_loc][:band_1_off + 1].count(chr)

        # Character not found in the band
        if band_0_t > band_1_t:
            return band_0, band_1, len(read) - i - 1 # Offset is i + 1 (matched till position i+1)
        
        # Position of the character in bwt which will become the new band
        if chr == "C":
            band_0_t += indices["A"][-1]
            band_1_t += indices["A"][-1]
        elif chr =="G":
            band_0_t += indices["A"][-1] + indices["C"][-1]
            band_1_t += indices["A"][-1] + indices["C"][-1]
        elif chr == "T":
            band_0_t += indices["A"][-1] + indices["C"][-1] + indices["G"][-1]
            band_1_t += indices["A"][-1] + indices["C"][-1] + indices["G"][-1]
        
        # New band for next character
        band_0 = band_0_t - 1
        band_1 = band_1_t - 1
    
    # If whole string matches, offset is zero
    return band_0, band_1, 0

def complementRead(read):
    """
    Input: string representing the read (including the newline("\n") at the end of the string)
    Output: string representing the complement of the read (without the newline)
    """
    # Replace with lower-case characters to avoid double replacing
    # ( "A"->"T", then later "T"->"A"
    #   "C"->"G", then later "G"->"C" )
    read = read.replace("A", "t").replace("C", "g").replace("G", "c").replace("T", "a")
    # Reverse the read because it is complement, and also change to upper-case
    return read[::-1].upper()

def countMismatches(s1, s2):
    """
    Input: two strings (representing the read and the corresponding portion of the reference)
    Output: Number of character mismatches in the two strings
    """
    return sum(a != b for a, b in zip(s1, s2))

def getLocsOfRead(read, bwt_string, indices, mapToReference, referenceSequence):
    """
    Input: read (string): The read string,
           bwt_string(list<string>): The last column of the BWT
           indices (dict<string, list<integer>>): The map of delta milestones
           mapToReference (list<integer>): The mapping from BWT to the reference string
           referenceSequence (list<string>): The reference string
    Output: list<integer>: list of possible positions where read matches in the reference string
    """
    # Keep track of matching positions
    positions = list()
    # Find matching band in BWT and its offset
    band_0, band_1, off = getBand(read, bwt_string, indices)
    print(band_0, band_1, off)

    # Check in the band to ensure 2 or less mismatches
    for i in range(band_0, band_1 + 1):
        # Position in the reference string
        pos = int(mapToReference[i]) - off 
        # Index and offset into the list of reference segments
        pos_idx = pos // 100
        pos_off = pos % 100
        # print(pos, pos_idx, pos_off)

        # Get the snippet of reference string at the matching location from the list of reference segments
        snip = referenceSequence[pos_idx][pos_off:-1] # Start from the index and offset calculater previously
        # Keep adding segments until the length is equal or more
        while len(snip) < len(read):
            pos_idx += 1
            # print(pos_idx)
            snip += referenceSequence[pos_idx][:-1]
        # Remove the extra characters
        snip = snip[:len(read)]

        # Count it as a match only if there are less than two mismatches
        if countMismatches(snip, read) <= 2:
            positions.append(pos)

    return positions

def MatchReadToLoc(read):
    """
    Input: a read (string)
    Output: list of potential locations at which the read may match the reference sequence. 
    Refer to example in the README file.
    IMPORTANT: This function must also handle the following:
        1. cases where the read does not match with the reference sequence
        2. any other special case that you may encounter
    """
    # function body - Begins

    # Get rid of the newline character ("\n") at the end of the read
    read = read[:-1]
    # Convert "N"s to "A"s
    read = read.replace("N", "A")

    # Compute the delta milestones if not already done,
    # and keep it global for reuse next time.
    global INDICES
    if INDICES is None:
        INDICES = getIndices(LastCol)

    # Get list of matching locations for the read
    positions = getLocsOfRead(read, LastCol, INDICES, Map, RefSeq)
    # print(positions)
    # Do the same for the complement of the read
    read_comp = complementRead(read)
    positions += getLocsOfRead(read_comp, LastCol, INDICES, Map, RefSeq)
    # print(positions)

    # function body - Ends
    return positions # list of potential locations at which the read may match the reference sequence.

def WhichExon(positions):
    """
    Input: list of potential locations at which the read may match the reference sequence.
    Output: Update(increment) to the counts of the 12 exons
    IMPORTANT: This function must also handle the following:
        1. cases where the read does not match with the reference sequence
        2. cases where there are more than one matches (match at two exons)
        3. any other special case that you may encounter
    """
    r1,r2,r3,r4,r5,r6,g1,g2,g3,g4,g5,g6 = 0,0,0,0,0,0,0,0,0,0,0,0
    # function body - Begins

    for i in positions:
        # To keep count of which red exon the position correspond to
        red_exons = np.zeros(6, dtype=int)
        # To keep count of which green exon the position correspond to
        green_exons = np.zeros(6, dtype=int)
        # Count matching exons :
        for j in range(6):
            if RedExonPos[j][0] <= i <= RedExonPos[j][1]:
                red_exons[j] = 1
            if GreenExonPos[j][0] <= i <= GreenExonPos[j][1]:
                green_exons[j] = 1
        
        # If the position corresponds to both a red and a green exon
        # then it contributes 0.5 to both exon
        if red_exons.sum() > 0 and green_exons.sum() > 0:
            if red_exons[0] > 0: r1 += 0.5
            if red_exons[1] > 0: r2 += 0.5
            if red_exons[2] > 0: r3 += 0.5
            if red_exons[3] > 0: r4 += 0.5
            if red_exons[4] > 0: r5 += 0.5
            if red_exons[5] > 0: r6 += 0.5
            if green_exons[0] > 0: g1 += 0.5
            if green_exons[1] > 0: g2 += 0.5
            if green_exons[2] > 0: g3 += 0.5
            if green_exons[3] > 0: g4 += 0.5
            if green_exons[4] > 0: g5 += 0.5
            if green_exons[5] > 0: g6 += 0.5
        # Otherwise it contributes 1 to whichever color exon it matches
        else:
            if red_exons[0] > 0: r1 += 1
            if red_exons[1] > 0: r2 += 1
            if red_exons[2] > 0: r3 += 1
            if red_exons[3] > 0: r4 += 1
            if red_exons[4] > 0: r5 += 1
            if red_exons[5] > 0: r6 += 1
            if green_exons[0] > 0: g1 += 1
            if green_exons[1] > 0: g2 += 1
            if green_exons[2] > 0: g3 += 1
            if green_exons[3] > 0: g4 += 1
            if green_exons[4] > 0: g5 += 1
            if green_exons[5] > 0: g6 += 1

    # function body - Ends
    return np.array([r1,r2,r3,r4,r5,r6,g1,g2,g3,g4,g5,g6])

def ComputeProb(ExonMatchCounts):
    """
    Input: The counts for each exon
    Output: Probabilities of each of the four configurations (a list of four real numbers)
    """
    # function body - Begins

    # Convert the ExonMatchCounts array to integer array
    # (for compatibility with built-in functions)
    matchCounts = np.array([i for i in ExonMatchCounts], dtype=int)
    # Total counts for each of the six exons (red + green)
    total_counts = np.array([matchCounts[i] + matchCounts[i + 6] for i in range(6)], dtype=int)

    # probability of match in the red exon for exons 2, 3, 4 and 5
    # r / (r + g)
    config_1_red = [0.33, 0.33, 0.33, 0.33] # L, M, M
    config_2_red = [0.5, 0.5, 0, 0] # L, ML, M
    config_3_red = [0.25, 0.25, 0.5, 0.5] # L, ML, M, M
    config_4_red = [0.25, 0.25, 0.25, 0.5] # ML, M, M

    # Probability of match in exon 2
    P0 = 0
    for i in range(len(config_1_red)):
        # P0 += ncr(total_counts[i + 1], matchCounts[i + 1]) * (config_1_red[i]**matchCounts[i + 1]) * ((1 - config_1_red[i])**matchCounts[i + 7])
        # The above line is hard to read, so here is a breakdown of the line
        prob_red_match = config_1_red[i] # Probability of matching in the L exon at position 2
        prob_green_match = 1 - prob_red_match # # Probability of matching in the M exon at position 2
        red_matches = matchCounts[i + 1] # Number of actual matches in the L exon at postion 2
        green_matches = matchCounts[i + 7] # Number of actual matches in the M exon at postion 2

        # Binomial distribution:
        # ∑ (nCr * p^r * (1-p)^s)
        # where n: total number of events
        #       p: probability of success
        #       r: number of events of success
        #       s: number of events of failure
        temp = math.comb(total_counts[i + 1], red_matches) # nCr
        temp *= (prob_red_match ** red_matches) # successful event (red match)
        temp *= (prob_green_match ** green_matches) # failure event (green match)
        P0 += temp # ∑
        # P0 += math.comb(total_counts[i + 1], red_matches) * (prob_red_match ** red_matches) * (prob_green_match ** green_matches)
    
    # Same logic for probability of match in exon 3
    P1 = 0
    for i in range(len(config_2_red)):
        prob_red_match = config_2_red[i]
        prob_green_match = 1 - prob_red_match
        red_matches = matchCounts[i + 1]
        green_matches = matchCounts[i + 7]
        P1 += math.comb(total_counts[i + 1], red_matches) * (prob_red_match ** red_matches) * (prob_green_match ** green_matches)
    
    # Same logic for probability of match in exon 4
    P2 = 0
    for i in range(len(config_3_red)):
        prob_red_match = config_3_red[i]
        prob_green_match = 1 - prob_red_match
        red_matches = matchCounts[i + 1]
        green_matches = matchCounts[i + 7]
        P2 += math.comb(total_counts[i + 1], red_matches) * (prob_red_match ** red_matches) * (prob_green_match ** green_matches)
    
    # Same logic for probability of match in exon 5
    P3 = 0
    for i in range(len(config_4_red)):
        prob_red_match = config_4_red[i]
        prob_green_match = 1 - prob_red_match
        red_matches = matchCounts[i + 1]
        green_matches = matchCounts[i + 7]
        P3 += math.comb(total_counts[i + 1], red_matches) * (prob_red_match ** red_matches) * (prob_green_match ** green_matches)

    # function body - ends
    return [P0, P1, P2, P3]

def BestMatch(ListProb):
    """
    Input: Probabilities of each of the four configurations (a list of four real numbers)
    Output: Most likely configuration (an integer). Refer to lecture slides
    """
    # function body - Begins

    # Consider the highest probability configuration as the possible configuration
    # What else to do?
    MostLikelyConfiguration = np.argmax(ListProb)

    # function body - ends
    return MostLikelyConfiguration # it holds 0, 1, 2, or 3

if __name__ == "__main__":
    t0 = time.time()
    # load all the data files
    LastCol = loadLastCol("../data/chrX_last_col.txt") # loads the last column
    RefSeq = loadRefSeq("../data/chrX.fa") # loads the reference sequence
    Reads = loadReads("../data/reads") # loads the reads
    Map = loadMapToRefSeq("../data/chrX_map.txt") # loads the mapping to the reference sequence
    
    # run the functions
    ExonMatchCounts = np.zeros(12) # initialize the counts for exons
    for i, read in enumerate(Reads): # update the counts for exons
        positions = MatchReadToLoc(read) # get the list of potential match locations
        tempu = WhichExon(positions)
        ExonMatchCounts += tempu # update the counts of exons, if applicable
    
    # print("Matches", ExonMatchCounts)
    ListProb = ComputeProb(ExonMatchCounts) # compute probabilities of each of the four configurations
    # print("Probabilities", ListProb
    MostLikely = BestMatch(ListProb) # find the most likely configuration
    print("Configuration %d is the best match"%MostLikely)

    tttt = time.time() - t0
    print("Time:", tttt//60, "minutes", tttt%60, "seconds")
