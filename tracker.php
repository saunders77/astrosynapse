<?php
declare(strict_types=1);

const ASTROSYNAPSE_REPORT_FILE = __DIR__ . '/astrosynapse_reports.jsonl';

function h($value): string
{
    return htmlspecialchars((string)$value, ENT_QUOTES | ENT_SUBSTITUTE, 'UTF-8');
}

function value_at($value, array $path, $default = '')
{
    $cursor = $value;
    foreach ($path as $part) {
        if (!is_array($cursor) || !array_key_exists($part, $cursor)) {
            return $default;
        }
        $cursor = $cursor[$part];
    }
    return $cursor;
}

function first_value($value, array $paths, $default = '')
{
    foreach ($paths as $path) {
        $found = value_at($value, $path, null);
        if ($found !== null && $found !== '') {
            return $found;
        }
    }
    return $default;
}

function format_time_value($value): string
{
    if (is_numeric($value) && (float)$value > 0) {
        return date('Y-m-d H:i:s', (int)$value);
    }
    if (is_string($value) && trim($value) !== '') {
        return $value;
    }
    return '-';
}

function format_seconds_value($value): string
{
    if (!is_numeric($value)) {
        return '-';
    }
    $seconds = max(0, (int)round((float)$value));
    $hours = intdiv($seconds, 3600);
    $seconds %= 3600;
    $minutes = intdiv($seconds, 60);
    $seconds %= 60;
    if ($hours > 0) {
        return sprintf('%dh %02dm %02ds', $hours, $minutes, $seconds);
    }
    if ($minutes > 0) {
        return sprintf('%dm %02ds', $minutes, $seconds);
    }
    return sprintf('%ds', $seconds);
}

function format_score($value): string
{
    if (!is_numeric($value)) {
        return '-';
    }
    return sprintf('%.1f%%', ((float)$value) * 100.0);
}

function read_reports(): array
{
    if (!is_readable(ASTROSYNAPSE_REPORT_FILE)) {
        return [];
    }

    $lines = file(ASTROSYNAPSE_REPORT_FILE, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    if (!is_array($lines)) {
        return [];
    }

    $entries = [];
    foreach ($lines as $index => $line) {
        $entry = json_decode($line, true);
        if (is_array($entry)) {
            $entry['_line'] = $index + 1;
            $entries[] = $entry;
        }
    }
    return $entries;
}

$entries = read_reports();
usort($entries, function (array $a, array $b): int {
    return (float)($b['received_at'] ?? 0) <=> (float)($a['received_at'] ?? 0);
});

$runFilter = trim((string)($_GET['run'] ?? ''));
if ($runFilter !== '') {
    $entries = array_values(array_filter($entries, function (array $entry) use ($runFilter): bool {
        $payload = $entry['payload'] ?? [];
        $runName = (string)first_value($payload, [
            ['run_name'],
            ['summary', 'run_name'],
        ], '');
        return strcasecmp($runName, $runFilter) === 0;
    }));
}

$limitParam = strtolower(trim((string)($_GET['limit'] ?? '200')));
$limit = $limitParam === 'all' ? count($entries) : max(1, min(1000, (int)$limitParam));
$visibleEntries = array_slice($entries, 0, $limit);

$latestByRun = [];
foreach ($entries as $entry) {
    $payload = $entry['payload'] ?? [];
    $runName = (string)first_value($payload, [
        ['run_name'],
        ['summary', 'run_name'],
    ], 'unknown');
    if (!isset($latestByRun[$runName])) {
        $latestByRun[$runName] = $entry;
    }
}

$uniqueRuns = count($latestByRun);
$latestEntry = $entries[0] ?? null;
$latestPayload = is_array($latestEntry) ? ($latestEntry['payload'] ?? []) : [];
?>
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>AstroSynapse Iteration Tracker</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #f6f8fb;
            --surface: #ffffff;
            --line: #d8dee8;
            --text: #162033;
            --muted: #5c687a;
            --accent: #2364aa;
            --good: #19734f;
            --warn: #9b5c00;
            --bad: #b42318;
        }
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            background: var(--bg);
            color: var(--text);
            font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        header,
        main {
            width: min(1480px, calc(100% - 32px));
            margin: 0 auto;
        }
        header {
            padding: 28px 0 18px;
        }
        h1 {
            margin: 0 0 6px;
            font-size: 28px;
            letter-spacing: 0;
        }
        .subtle {
            color: var(--muted);
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 16px 0 18px;
        }
        .stat {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 14px;
        }
        .stat strong {
            display: block;
            font-size: 24px;
            line-height: 1.1;
        }
        .toolbar {
            display: flex;
            align-items: end;
            flex-wrap: wrap;
            gap: 12px;
            margin: 0 0 16px;
        }
        label {
            display: grid;
            gap: 4px;
            color: var(--muted);
            font-size: 12px;
        }
        input {
            min-width: 190px;
            border: 1px solid var(--line);
            border-radius: 6px;
            padding: 8px 10px;
            font: inherit;
        }
        button,
        .button {
            border: 1px solid var(--accent);
            border-radius: 6px;
            background: var(--accent);
            color: #fff;
            padding: 8px 12px;
            font: inherit;
            text-decoration: none;
            cursor: pointer;
        }
        section {
            margin: 0 0 22px;
        }
        h2 {
            margin: 0 0 10px;
            font-size: 18px;
        }
        .table-wrap {
            overflow-x: auto;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            min-width: 1120px;
        }
        th,
        td {
            border-bottom: 1px solid var(--line);
            padding: 9px 10px;
            text-align: left;
            vertical-align: top;
            white-space: nowrap;
        }
        th {
            background: #eef3f9;
            color: #27344a;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: .04em;
        }
        tr:last-child td {
            border-bottom: 0;
        }
        .tag {
            display: inline-block;
            border-radius: 999px;
            padding: 2px 8px;
            background: #e8eef7;
            color: #24344d;
            font-size: 12px;
        }
        .promoted {
            background: #e4f5ee;
            color: var(--good);
        }
        .reset {
            background: #fff4df;
            color: var(--warn);
        }
        details {
            white-space: normal;
        }
        summary {
            color: var(--accent);
            cursor: pointer;
        }
        pre {
            max-width: 760px;
            max-height: 420px;
            overflow: auto;
            margin: 8px 0 0;
            padding: 10px;
            border: 1px solid var(--line);
            border-radius: 6px;
            background: #0f1724;
            color: #edf3ff;
            font-size: 12px;
            white-space: pre-wrap;
        }
        .empty {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 22px;
        }
        @media (max-width: 720px) {
            header,
            main {
                width: min(100% - 20px, 1480px);
            }
            h1 {
                font-size: 24px;
            }
            .toolbar {
                align-items: stretch;
            }
            input,
            button,
            .button {
                width: 100%;
            }
        }
    </style>
</head>
<body>
<header>
    <h1>AstroSynapse Iteration Tracker</h1>
    <div class="subtle">
        Reading <?php echo h(basename(ASTROSYNAPSE_REPORT_FILE)); ?> from the same directory as this page.
    </div>
    <div class="stats">
        <div class="stat">
            <strong><?php echo h((string)count($entries)); ?></strong>
            Stored reports<?php echo $runFilter !== '' ? ' for this run' : ''; ?>
        </div>
        <div class="stat">
            <strong><?php echo h((string)$uniqueRuns); ?></strong>
            Run<?php echo $uniqueRuns === 1 ? '' : 's'; ?> seen
        </div>
        <div class="stat">
            <strong><?php echo h(first_value($latestPayload, [['iteration']], '-')); ?></strong>
            Latest iteration
        </div>
        <div class="stat">
            <strong><?php echo h(format_time_value($latestEntry['received_at'] ?? '')); ?></strong>
            Latest report received
        </div>
    </div>
    <form class="toolbar" method="get">
        <label>
            Run filter
            <input name="run" value="<?php echo h($runFilter); ?>" placeholder="example: default">
        </label>
        <label>
            Limit
            <input name="limit" value="<?php echo h($limitParam); ?>" placeholder="200 or all">
        </label>
        <button type="submit">Apply</button>
        <a class="button" href="<?php echo h(strtok($_SERVER['REQUEST_URI'] ?? 'tracker.php', '?') ?: 'tracker.php'); ?>">Clear</a>
    </form>
</header>
<main>
    <?php if ($entries === []): ?>
        <div class="empty">
            No reports have been received yet. Once training posts to <code>report.php</code>, entries will appear here.
        </div>
    <?php else: ?>
        <section>
            <h2>Latest By Run</h2>
            <div class="table-wrap">
                <table>
                    <thead>
                    <tr>
                        <th>Run</th>
                        <th>Iteration</th>
                        <th>Action</th>
                        <th>Score</th>
                        <th>Eval</th>
                        <th>Elo</th>
                        <th>Completed</th>
                        <th>Received</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($latestByRun as $runName => $entry): ?>
                        <?php
                        $payload = $entry['payload'] ?? [];
                        $action = (string)first_value($payload, [['action'], ['promotion_evaluation', 'summary', 'action']], '-');
                        $promoted = (bool)first_value($payload, [['outcome', 'promoted'], ['promotion_evaluation', 'summary', 'promoted']], false);
                        $wins = first_value($payload, [['promotion_evaluation', 'summary', 'wins']], 0);
                        $losses = first_value($payload, [['promotion_evaluation', 'summary', 'losses']], 0);
                        $games = first_value($payload, [['promotion_evaluation', 'games_completed'], ['promotion_evaluation', 'summary', 'games_played']], 0);
                        ?>
                        <tr>
                            <td><?php echo h($runName); ?></td>
                            <td><?php echo h(first_value($payload, [['iteration']], '-')); ?></td>
                            <td>
                                <span class="tag <?php echo $promoted ? 'promoted' : ($action === 'reset_candidate' ? 'reset' : ''); ?>">
                                    <?php echo h($action); ?>
                                </span>
                            </td>
                            <td><?php echo h(format_score(first_value($payload, [['outcome', 'candidate_score'], ['promotion_evaluation', 'summary', 'score']], null))); ?></td>
                            <td><?php echo h($wins . '-' . $losses . ' / ' . $games); ?></td>
                            <td><?php echo h(first_value($payload, [['outcome', 'current_elo'], ['summary', 'current_elo']], '-') . ' / ' . first_value($payload, [['outcome', 'best_elo'], ['summary', 'best_elo']], '-')); ?></td>
                            <td><?php echo h(format_time_value(first_value($payload, [['iteration_completed_at'], ['generated_at']], ''))); ?></td>
                            <td><?php echo h(format_time_value($entry['received_at'] ?? '')); ?></td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
        </section>

        <section>
            <h2>Recent Iterations</h2>
            <div class="table-wrap">
                <table>
                    <thead>
                    <tr>
                        <th>Received</th>
                        <th>Completed</th>
                        <th>Run</th>
                        <th>Iter</th>
                        <th>Action</th>
                        <th>Score</th>
                        <th>Eval</th>
                        <th>Training</th>
                        <th>Samples</th>
                        <th>Elo</th>
                        <th>Duration</th>
                        <th>Host</th>
                        <th>Full Payload</th>
                    </tr>
                    </thead>
                    <tbody>
                    <?php foreach ($visibleEntries as $entry): ?>
                        <?php
                        $payload = $entry['payload'] ?? [];
                        $runName = first_value($payload, [['run_name'], ['summary', 'run_name']], 'unknown');
                        $action = (string)first_value($payload, [['action'], ['promotion_evaluation', 'summary', 'action']], '-');
                        $promoted = (bool)first_value($payload, [['outcome', 'promoted'], ['promotion_evaluation', 'summary', 'promoted']], false);
                        $wins = first_value($payload, [['promotion_evaluation', 'summary', 'wins']], 0);
                        $losses = first_value($payload, [['promotion_evaluation', 'summary', 'losses']], 0);
                        $evalGames = first_value($payload, [['promotion_evaluation', 'games_completed'], ['promotion_evaluation', 'summary', 'games_played']], 0);
                        $trainingGames = first_value($payload, [['training', 'games_completed']], 0);
                        $trainingTarget = first_value($payload, [['training', 'games_target']], 0);
                        $matches = first_value($payload, [['training', 'matches_completed']], 0);
                        $matchesTarget = first_value($payload, [['training', 'matches_target']], 0);
                        $samples = first_value($payload, [['training', 'samples_collected']], 0);
                        $sampleCandidates = first_value($payload, [['training', 'sample_candidates']], 0);
                        $currentElo = first_value($payload, [['outcome', 'current_elo'], ['summary', 'current_elo']], '-');
                        $bestElo = first_value($payload, [['outcome', 'best_elo'], ['summary', 'best_elo']], '-');
                        $json = json_encode($entry, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
                        ?>
                        <tr>
                            <td><?php echo h(format_time_value($entry['received_at'] ?? '')); ?></td>
                            <td><?php echo h(format_time_value(first_value($payload, [['iteration_completed_at'], ['generated_at']], ''))); ?></td>
                            <td><?php echo h($runName); ?></td>
                            <td><?php echo h(first_value($payload, [['iteration']], '-')); ?></td>
                            <td>
                                <span class="tag <?php echo $promoted ? 'promoted' : ($action === 'reset_candidate' ? 'reset' : ''); ?>">
                                    <?php echo h($action); ?>
                                </span>
                            </td>
                            <td><?php echo h(format_score(first_value($payload, [['outcome', 'candidate_score'], ['promotion_evaluation', 'summary', 'score']], null))); ?></td>
                            <td><?php echo h($wins . '-' . $losses . ' / ' . $evalGames); ?></td>
                            <td><?php echo h($trainingGames . ' / ' . $trainingTarget . ' games, ' . $matches . ' / ' . $matchesTarget . ' matches'); ?></td>
                            <td><?php echo h($samples . ' kept, ' . $sampleCandidates . ' candidates'); ?></td>
                            <td><?php echo h($currentElo . ' / ' . $bestElo); ?></td>
                            <td><?php echo h(format_seconds_value(first_value($payload, [['iteration_duration_seconds']], null))); ?></td>
                            <td><?php echo h(first_value($payload, [['host']], '-')); ?></td>
                            <td>
                                <details>
                                    <summary>View JSON</summary>
                                    <pre><?php echo h(is_string($json) ? $json : '{}'); ?></pre>
                                </details>
                            </td>
                        </tr>
                    <?php endforeach; ?>
                    </tbody>
                </table>
            </div>
            <p class="subtle">
                Showing <?php echo h((string)count($visibleEntries)); ?> of <?php echo h((string)count($entries)); ?> report(s).
            </p>
        </section>
    <?php endif; ?>
</main>
</body>
</html>
