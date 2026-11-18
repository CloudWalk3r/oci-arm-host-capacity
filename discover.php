<?php
declare(strict_types=1);

require __DIR__ . '/vendor/autoload.php';

use Dotenv\Dotenv;
use Hitrov\OciApi;
use Hitrov\OciConfig;

$envFilename = empty($argv[1]) ? '.env' : $argv[1];
$dotenv = Dotenv::createUnsafeImmutable(__DIR__, $envFilename);
$dotenv->safeLoad();

$config = new OciConfig(
    getenv('OCI_REGION'),
    getenv('OCI_USER_ID'),
    getenv('OCI_TENANCY_ID'),
    getenv('OCI_KEY_FINGERPRINT'),
    getenv('OCI_PRIVATE_KEY_FILENAME'),
    getenv('OCI_AVAILABILITY_DOMAIN') ?: null,
    getenv('OCI_SUBNET_ID'),
    getenv('OCI_IMAGE_ID'),
    (int) getenv('OCI_OCPUS'),
    (int) getenv('OCI_MEMORY_IN_GBS')
);

$shape = getenv('OCI_SHAPE');
$region = getenv('OCI_REGION');
$tenancy = getenv('OCI_TENANCY_ID');
$iaas = "https://iaas.$region.oraclecloud.com/20160918";

$api = new OciApi();

function out(string $label, $data): void {
    echo "===== $label =====\n";
    echo json_encode($data, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES) . "\n\n";
}

// 1) VCNs
try {
    $vcns = $api->call($config, "$iaas/vcns/", 'GET', null, ['compartmentId' => $tenancy]);
    $vcnSummary = array_map(fn($v) => [
        'id' => $v['id'],
        'displayName' => $v['displayName'] ?? null,
        'lifecycleState' => $v['lifecycleState'] ?? null,
    ], $vcns);
    out('VCNs', $vcnSummary);
} catch (Throwable $e) {
    out('VCNs ERROR', $e->getMessage());
}

// 2) Subnets
try {
    $subnets = $api->call($config, "$iaas/subnets/", 'GET', null, ['compartmentId' => $tenancy]);
    $subnetSummary = array_map(fn($s) => [
        'id' => $s['id'],
        'displayName' => $s['displayName'] ?? null,
        'vcnId' => $s['vcnId'] ?? null,
        'prohibitPublicIpOnVnic' => $s['prohibitPublicIpOnVnic'] ?? null,
        'lifecycleState' => $s['lifecycleState'] ?? null,
    ], $subnets);
    out('SUBNETS', $subnetSummary);
} catch (Throwable $e) {
    out('SUBNETS ERROR', $e->getMessage());
}

// 3) Images for the shape
try {
    $images = $api->call($config, "$iaas/images/", 'GET', null, [
        'compartmentId' => $tenancy,
        'shape' => $shape,
        'sortBy' => 'TIMECREATED',
        'sortOrder' => 'DESC',
    ]);
    $imageSummary = array_map(fn($i) => [
        'id' => $i['id'],
        'displayName' => $i['displayName'] ?? null,
        'operatingSystem' => $i['operatingSystem'] ?? null,
        'operatingSystemVersion' => $i['operatingSystemVersion'] ?? null,
    ], $images);
    out("IMAGES for $shape", $imageSummary);
} catch (Throwable $e) {
    out('IMAGES ERROR', $e->getMessage());
}
