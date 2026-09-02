// Recompute search-log aggregates in Go and check them against README.md.
//
// Covers: candidate counts, training minutes, best fitness/accuracy, seed
// comparison, budget figures, and generation improvement tracking.
//
// Build: go build -o verify/gocheck/gocheck verify/gocheck
// Run:   verify/gocheck/gocheck [root]

package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type row struct {
	gen        string
	cand       string
	mutation   string
	params     int
	macs       int
	peakAct    int
	acc        float64
	fitness    float64
	deployable int
	trainS     float64
	genome     string
}

func main() {
	root := "."
	if len(os.Args) > 1 {
		root = os.Args[1]
	}

	rows := loadCSV(filepath.Join(root, "results", "search_log.csv"))
	readme, err := os.ReadFile(filepath.Join(root, "README.md"))
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot read README.md: %v\n", err)
		os.Exit(1)
	}
	readmeStr := string(readme)

	var ok []row
	for _, r := range rows {
		if r.deployable == 1 {
			ok = append(ok, r)
		}
	}
	seed := rows[0]
	best := ok[0]
	for _, r := range ok[1:] {
		if r.fitness > best.fitness {
			best = r
		}
	}

	rejected := 0
	totalTrainS := 0.0
	for _, r := range rows {
		if r.deployable == 0 {
			rejected++
		}
		totalTrainS += r.trainS
	}
	totalTrainMin := int(math.Round(totalTrainS / 60.0))

	// Duplicate genomes.
	genomeCounts := map[string]int{}
	for _, r := range rows {
		genomeCounts[r.genome]++
	}
	dupCount := 0
	for _, c := range genomeCounts {
		if c > 1 {
			dupCount++
		}
	}

	// Duplicate training minutes.
	seenGenomes := map[string]bool{}
	dupTrainS := 0.0
	for _, r := range rows {
		if seenGenomes[r.genome] {
			dupTrainS += r.trainS
		} else {
			seenGenomes[r.genome] = true
		}
	}
	dupTrainMin := int(math.Round(dupTrainS / 60.0))

	// Insert operator.
	insertCount := 0
	insertTrainable := 0
	for _, r := range rows {
		if strings.HasPrefix(r.mutation, "insert") {
			insertCount++
			insertTrainable += r.deployable
		}
	}

	gapPoints := 100.0 * (best.acc - seed.acc)
	paramDiff := best.params - seed.params
	macPct := 100.0 * (1.0 - float64(best.macs)/float64(seed.macs))

	bestKB := float64(best.params+best.peakAct) / 1024.0
	worstKB := 0.0
	for _, r := range ok {
		kb := float64(r.params+r.peakAct) / 1024.0
		if kb > worstKB {
			worstKB = kb
		}
	}
	headroom := 250.0 / worstKB

	moreAcc := 0
	for _, r := range ok {
		if r.acc > best.acc {
			moreAcc++
		}
	}
	accRank := moreAcc + 1

	// Which generations improved.
	bestSoFar := math.Inf(-1)
	var improved []string
	for _, r := range ok {
		if r.gen != "0" && r.fitness > bestSoFar {
			found := false
			for _, g := range improved {
				if g == r.gen {
					found = true
					break
				}
			}
			if !found {
				improved = append(improved, r.gen)
			}
		}
		if r.fitness > bestSoFar {
			bestSoFar = r.fitness
		}
	}

	var improvedStr string
	if len(improved) >= 2 {
		improvedStr = strings.Join(improved[:len(improved)-1], ", ") + " and " + improved[len(improved)-1]
	} else if len(improved) == 1 {
		improvedStr = improved[0]
	}
	stalled := 8 - len(improved)

	want := []string{
		fmt.Sprintf("%d candidates", len(rows)),
		fmt.Sprintf("%d child slots", len(rows)-1),
		fmt.Sprintf("%d rejected", rejected),
		fmt.Sprintf("%d evaluated twice", dupCount),
		fmt.Sprintf("%d minutes of training", totalTrainMin),
		fmt.Sprintf("%d minutes of duplicated", dupTrainMin),
		fmt.Sprintf("%d trained candidates", len(ok)),
		fmt.Sprintf("drawn %d times and produced %d trainable", insertCount, insertTrainable),
		fmt.Sprintf("**%.4f**", best.acc),
		fmt.Sprintf("**%.4f**", best.fitness),
		fmt.Sprintf("%.1f accuracy points", gapPoints),
		fmt.Sprintf("%d more", paramDiff),
		fmt.Sprintf("%.1f%% fewer", macPct),
		fmt.Sprintf("generation %s, candidate %s", best.gen, best.cand),
		fmt.Sprintf("%.1f KB", bestKB),
		fmt.Sprintf("%.1f KB", worstKB),
		fmt.Sprintf("%.1fx", headroom),
		fmt.Sprintf("%dth by accuracy", accRank),
		improvedStr,
		fmt.Sprintf("%d of the 8 generations", stalled),
	}

	failures := 0
	for _, x := range want {
		hit := strings.Contains(readmeStr, x)
		tag := "ok"
		if !hit {
			tag = "FAIL"
			failures++
		}
		fmt.Printf("  %-4s %s\n", tag, x)
	}

	if failures > 0 {
		fmt.Printf("%d of %d figures are not in README.md as written\n", failures, len(want))
		os.Exit(1)
	}
	fmt.Printf("Go reproduces all %d figures from the search log\n", len(want))
}

func loadCSV(path string) []row {
	f, err := os.Open(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot open %s: %v\n", path, err)
		os.Exit(1)
	}
	defer f.Close()

	r := csv.NewReader(f)
	records, err := r.ReadAll()
	if err != nil {
		fmt.Fprintf(os.Stderr, "CSV parse error: %v\n", err)
		os.Exit(1)
	}

	var rows []row
	for _, rec := range records[1:] { // skip header
		params, _ := strconv.Atoi(rec[4])
		macs, _ := strconv.Atoi(rec[5])
		peakAct, _ := strconv.Atoi(rec[6])
		acc, _ := strconv.ParseFloat(rec[7], 64)
		var fitness float64
		if rec[8] == "-inf" {
			fitness = math.Inf(-1)
		} else {
			fitness, _ = strconv.ParseFloat(rec[8], 64)
		}
		deployable, _ := strconv.Atoi(rec[9])
		trainS, _ := strconv.ParseFloat(rec[10], 64)
		rows = append(rows, row{
			gen:        rec[0],
			cand:       rec[1],
			mutation:   rec[3],
			params:     params,
			macs:       macs,
			peakAct:    peakAct,
			acc:        acc,
			fitness:    fitness,
			deployable: deployable,
			trainS:     trainS,
			genome:     rec[11],
		})
	}
	return rows
}
