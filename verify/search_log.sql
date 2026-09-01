-- Recompute the search summary that README.md quotes, from results/search_log.csv.
--
-- Every figure in the README is derived by check_numbers.py, in Python, from
-- the same file that the search wrote with Python. If that aggregation were
-- wrong, nothing would notice: the checker and the thing it checks share an
-- implementation. This derives the same figures in SQL and nothing else, so a
-- mistake would have to be made identically in both to survive.
--
-- Each row of output is a string that must appear verbatim in README.md.
-- verify/verify.sh does that matching.
--
-- Run: sqlite3 -init verify/search_log.sql :memory: "" < /dev/null

.mode csv
.headers off
.import --csv results/search_log.csv log

-- The columns arrive as text. Cast once here rather than at every use.
CREATE TEMP VIEW rows_t AS
    SELECT rowid AS rid, gen, cand, mutation, genome,
           CAST(params AS INT)   AS params,
           CAST(macs AS INT)     AS macs,
           CAST(peak_act AS INT) AS peak_act,
           CAST(acc AS REAL)     AS acc,
           CAST(fitness AS REAL) AS fitness,
           CAST(train_s AS REAL) AS train_s,
           CAST(deployable AS INT) AS deployable
    FROM log;

CREATE TEMP VIEW ok AS SELECT * FROM rows_t WHERE deployable = 1;
CREATE TEMP VIEW seed AS SELECT * FROM rows_t ORDER BY rid LIMIT 1;
CREATE TEMP VIEW best AS SELECT * FROM ok ORDER BY fitness DESC LIMIT 1;

-- A genome that appears again is a re-evaluation. The first sighting was work,
-- so only the later ones count as waste.
CREATE TEMP VIEW dup AS
    SELECT * FROM rows_t r
    WHERE EXISTS (SELECT 1 FROM rows_t p WHERE p.genome = r.genome AND p.rid < r.rid);

-- A generation improved if one of its deployable children beat the best
-- fitness seen in any earlier row of the log.
CREATE TEMP VIEW improved AS
    SELECT DISTINCT gen FROM
        (SELECT gen, fitness,
                MAX(fitness) OVER (ORDER BY rid
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev
         FROM ok)
    WHERE gen <> '0' AND (prev IS NULL OR fitness > prev)
    ORDER BY gen;

.mode list
.separator "|"

SELECT printf('%d candidates', (SELECT COUNT(*) FROM rows_t))
UNION ALL SELECT printf('%d child slots', (SELECT COUNT(*) - 1 FROM rows_t))
UNION ALL SELECT printf('%d rejected', (SELECT COUNT(*) FROM rows_t WHERE deployable = 0))
UNION ALL SELECT printf('%d evaluated twice',
    (SELECT COUNT(*) FROM (SELECT genome FROM rows_t GROUP BY genome HAVING COUNT(*) > 1)))
UNION ALL SELECT printf('%d minutes of training',
    (SELECT CAST(ROUND(SUM(train_s) / 60.0) AS INT) FROM rows_t))
UNION ALL SELECT printf('%d minutes of duplicated',
    (SELECT CAST(ROUND(SUM(train_s) / 60.0) AS INT) FROM dup))
UNION ALL SELECT printf('%d trained candidates', (SELECT COUNT(*) FROM ok))

-- The insert operator: drawn, and how many of those were trainable at all.
UNION ALL SELECT (SELECT printf('drawn %d times and produced %d trainable',
                                COUNT(*), SUM(deployable))
                  FROM rows_t WHERE mutation LIKE 'insert%')

-- The result table in section 4.
UNION ALL SELECT (SELECT printf('| seed, hand written | %.4f | %d,%03d | %d,%03d,%03d | %d,%03d | %.4f |',
                                acc, params / 1000, params % 1000,
                                macs / 1000000, (macs / 1000) % 1000, macs % 1000,
                                peak_act / 1000, peak_act % 1000, fitness) FROM seed)
UNION ALL SELECT (SELECT printf('**%.4f**', acc) FROM best)
UNION ALL SELECT (SELECT printf('| %d,%03d |', params / 1000, params % 1000) FROM best)
UNION ALL SELECT (SELECT printf('**%d,%03d,%03d**', macs / 1000000, (macs / 1000) % 1000,
                                macs % 1000) FROM best)
UNION ALL SELECT (SELECT printf('**%.4f**', fitness) FROM best)
UNION ALL SELECT printf('%.1f accuracy points',
    100 * ((SELECT acc FROM best) - (SELECT acc FROM seed)))
UNION ALL SELECT printf('%d more', (SELECT params FROM best) - (SELECT params FROM seed))
UNION ALL SELECT printf('%.1f%% fewer',
    100 * (1.0 - (SELECT macs FROM best) * 1.0 / (SELECT macs FROM seed)))
UNION ALL SELECT (SELECT printf('generation %s, candidate %s', gen, cand) FROM best)

-- Section 5's budget paragraph. The working set is weights plus one peak
-- activation, one byte each at int8.
UNION ALL SELECT (SELECT printf('%.1f KB', (params + peak_act) / 1024.0) FROM best)
UNION ALL SELECT (SELECT printf('%.1f KB', MAX((params + peak_act) / 1024.0)) FROM ok)
UNION ALL SELECT (SELECT printf('%.1fx', 250.0 / MAX((params + peak_act) / 1024.0)) FROM ok)

-- Section 8: the winner was not the most accurate candidate.
UNION ALL SELECT printf('%dth by accuracy',
    (SELECT COUNT(*) FROM ok WHERE acc > (SELECT acc FROM best)) + 1)
UNION ALL SELECT (SELECT printf('%.4f at %d,%03d,%03d', acc, macs / 1000000,
                                (macs / 1000) % 1000, macs % 1000)
                  FROM ok WHERE acc > (SELECT acc FROM best) ORDER BY acc DESC LIMIT 1)

-- Which generations beat everything seen before them, in file order.
UNION ALL SELECT (SELECT printf('%s and %s',
                        substr(all_g, 1, length(all_g) - length(last_g) - 2), last_g)
                  FROM (SELECT (SELECT GROUP_CONCAT(gen, ', ') FROM improved) AS all_g,
                               (SELECT MAX(gen) FROM improved) AS last_g))
UNION ALL SELECT printf('%d of the 8 generations',
                        8 - (SELECT COUNT(*) FROM improved));
