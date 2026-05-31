<?php
declare(strict_types=1);

header('Content-Type: application/json; charset=utf-8');
header('Cache-Control: no-store');

const ASTROSYNAPSE_REPORT_FILE = __DIR__ . '/astrosynapse_reports.jsonl';
const ASTROSYNAPSE_MAX_BODY_BYTES = 1048576;

function respond_json(int $status, array $payload): void
{
    http_response_code($status);
    echo json_encode($payload, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
    exit;
}

function optional_report_token(): string
{
    $envToken = getenv('ASTROSYNAPSE_REPORT_TOKEN');
    if (is_string($envToken) && trim($envToken) !== '') {
        return trim($envToken);
    }

    $tokenFile = __DIR__ . '/report_token.txt';
    if (is_readable($tokenFile)) {
        $fileToken = file_get_contents($tokenFile);
        if (is_string($fileToken) && trim($fileToken) !== '') {
            return trim($fileToken);
        }
    }

    return '';
}

function new_report_id(): string
{
    try {
        return bin2hex(random_bytes(12));
    } catch (Throwable $exc) {
        return str_replace('.', '', uniqid('', true));
    }
}

function is_assoc_array(array $value): bool
{
    if ($value === []) {
        return false;
    }
    return array_keys($value) !== range(0, count($value) - 1);
}

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    header('Allow: POST, OPTIONS');
    respond_json(204, ['ok' => true]);
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    header('Allow: POST, OPTIONS');
    respond_json(405, ['ok' => false, 'error' => 'Use POST with a JSON body.']);
}

$expectedToken = optional_report_token();
if ($expectedToken !== '') {
    $providedToken = $_SERVER['HTTP_X_ASTROSYNAPSE_TOKEN'] ?? ($_GET['token'] ?? '');
    if (!is_string($providedToken) || !hash_equals($expectedToken, $providedToken)) {
        respond_json(403, ['ok' => false, 'error' => 'Invalid report token.']);
    }
}

$contentLength = (int)($_SERVER['CONTENT_LENGTH'] ?? 0);
if ($contentLength > ASTROSYNAPSE_MAX_BODY_BYTES) {
    respond_json(413, ['ok' => false, 'error' => 'Request body is too large.']);
}

$rawBody = file_get_contents('php://input');
if (!is_string($rawBody)) {
    respond_json(400, ['ok' => false, 'error' => 'Unable to read request body.']);
}
if (strlen($rawBody) > ASTROSYNAPSE_MAX_BODY_BYTES) {
    respond_json(413, ['ok' => false, 'error' => 'Request body is too large.']);
}

$payload = json_decode($rawBody, true);
if (!is_array($payload) && $_POST !== []) {
    $payload = $_POST;
}
if (!is_array($payload) || !is_assoc_array($payload)) {
    respond_json(400, ['ok' => false, 'error' => 'Expected a JSON object payload.']);
}

$receivedAt = microtime(true);
$entry = [
    'id' => new_report_id(),
    'received_at' => $receivedAt,
    'received_datetime' => date('c', (int)$receivedAt),
    'remote_addr' => $_SERVER['REMOTE_ADDR'] ?? '',
    'user_agent' => $_SERVER['HTTP_USER_AGENT'] ?? '',
    'payload' => $payload,
];

$encoded = json_encode($entry, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
if (!is_string($encoded)) {
    respond_json(400, ['ok' => false, 'error' => 'Payload could not be encoded for storage.']);
}

$bytesWritten = @file_put_contents(ASTROSYNAPSE_REPORT_FILE, $encoded . PHP_EOL, FILE_APPEND | LOCK_EX);
if ($bytesWritten === false) {
    respond_json(500, [
        'ok' => false,
        'error' => 'Unable to write report data. Confirm this directory is writable by PHP.',
    ]);
}

respond_json(200, [
    'ok' => true,
    'id' => $entry['id'],
    'received_at' => $entry['received_at'],
    'received_datetime' => $entry['received_datetime'],
]);
