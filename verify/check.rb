# Recompute search-log aggregates in Ruby and check them against README.md.
#
# Covers candidate counts, training minutes, best fitness/accuracy, seed
# comparison, budget figures, and generation improvement tracking.

require "csv"

root = ARGV[0] || "."

rows = CSV.read(File.join(root, "results", "search_log.csv"), headers: true).map { |r|
  {
    gen:        r["gen"],
    cand:       r["cand"],
    mutation:   r["mutation"],
    params:     r["params"].to_i,
    macs:       r["macs"].to_i,
    peak_act:   r["peak_act"].to_i,
    acc:        r["acc"].to_f,
    fitness:    r["fitness"] == "-inf" ? -Float::INFINITY : r["fitness"].to_f,
    deployable: r["deployable"].to_i,
    train_s:    r["train_s"].to_f,
    genome:     r["genome"],
  }
}

readme = File.read(File.join(root, "README.md"), encoding: "utf-8")

ok   = rows.select { |r| r[:deployable] == 1 }
seed = rows[0]
best = ok.max_by { |r| r[:fitness] }

rejected = rows.count { |r| r[:deployable] == 0 }

genome_counts = rows.group_by { |r| r[:genome] }
dup_count = genome_counts.count { |_, v| v.size > 1 }

total_train_min = (rows.sum { |r| r[:train_s] } / 60.0).round
seen = {}
dup_train_s = 0.0
rows.each do |r|
  if seen.key?(r[:genome])
    dup_train_s += r[:train_s]
  else
    seen[r[:genome]] = true
  end
end
dup_train_min = (dup_train_s / 60.0).round

insert_rows = rows.select { |r| r[:mutation].start_with?("insert") }
insert_trainable = insert_rows.sum { |r| r[:deployable] }

gap_points = 100.0 * (best[:acc] - seed[:acc])
param_diff = best[:params] - seed[:params]
mac_pct = 100.0 * (1.0 - best[:macs].to_f / seed[:macs])

best_kb = (best[:params] + best[:peak_act]) / 1024.0
worst_kb = ok.map { |r| (r[:params] + r[:peak_act]) / 1024.0 }.max
headroom = 250.0 / worst_kb

more_acc = ok.select { |r| r[:acc] > best[:acc] }
acc_rank = more_acc.size + 1

# Which generations improved.
best_so_far = -Float::INFINITY
improved = []
ok.each do |r|
  if r[:gen] != "0" && r[:fitness] > best_so_far
    improved << r[:gen] unless improved.include?(r[:gen])
  end
  best_so_far = r[:fitness] if r[:fitness] > best_so_far
end

# Format the improved-generations string the way README has it.
if improved.size >= 2
  improved_str = improved[0..-2].join(", ") + " and " + improved[-1]
else
  improved_str = improved.join(", ")
end
stalled = 8 - improved.size

want = [
  "#{rows.size} candidates",
  "#{rows.size - 1} child slots",
  "#{rejected} rejected",
  "#{dup_count} evaluated twice",
  "#{total_train_min} minutes of training",
  "#{dup_train_min} minutes of duplicated",
  "#{ok.size} trained candidates",
  "drawn #{insert_rows.size} times and produced #{insert_trainable} trainable",
  "**#{format('%.4f', best[:acc])}**",
  "**#{format('%.4f', best[:fitness])}**",
  "#{format('%.1f', gap_points)} accuracy points",
  "#{param_diff} more",
  "#{format('%.1f', mac_pct)}% fewer",
  "generation #{best[:gen]}, candidate #{best[:cand]}",
  "#{format('%.1f', best_kb)} KB",
  "#{format('%.1f', worst_kb)} KB",
  "#{format('%.1f', headroom)}x",
  "#{acc_rank}th by accuracy",
  improved_str,
  "#{stalled} of the 8 generations",
]

failures = 0
want.each do |x|
  hit = readme.include?(x)
  tag = hit ? "ok" : "FAIL"
  puts "  %-4s %s" % [tag, x]
  failures += 1 unless hit
end

if failures > 0
  puts "#{failures} of #{want.size} figures are not in README.md as written"
  exit 1
end
puts "Ruby reproduces all #{want.size} figures from the search log"
