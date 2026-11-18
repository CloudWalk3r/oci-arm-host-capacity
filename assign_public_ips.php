<?php
declare(strict_types=1);

error_reporting(E_ALL & ~E_DEPRECATED);
require __DIR__ . '/vendor/autoload.php';

use Dotenv\Dotenv;
use Hitrov\OciApi;
use Hitrov\OciConfig;

$envFilename = empty($argv[1]) ? '.env' : $argv[1];
$dotenv = Dotenv::createUnsafeImmutable(__DIR__, $envFilename);
$dotenv->safeLoad();

$config = new OciConfig(
    getenv('OCI_REGION'), getenv('OCI_USER_ID'), getenv('OCI_TENANCY_ID'),
    getenv('OCI_KEY_FINGERPRINT'), getenv('OCI_PRIVATE_KEY_FILENAME'),
    getenv('OCI_AVAILABILITY_DOMAIN') ?: null, getenv('OCI_SUBNET_ID'),
    getenv('OCI_IMAGE_ID'), (int) getenv('OCI_OCPUS'), (int) getenv('OCI_MEMORY_IN_GBS')
);

$region = getenv('OCI_REGION');
$tenancy = getenv('OCI_TENANCY_ID');
$iaas = "https://iaas.$region.oraclecloud.com/20160918";
$api = new OciApi();

$instances = $api->getInstances($config);
foreach ($instances as $inst) {
    if (($inst['lifecycleState'] ?? '') === 'TERMINATED') continue;
    $name = $inst['displayName'];
    $instanceId = $inst['id'];

    // Wait for RUNNING
    for ($i = 0; $i < 30; $i++) {
        $cur = $api->call($config, "$iaas/instances/$instanceId", 'GET');
        if (($cur['lifecycleState'] ?? '') === 'RUNNING') break;
        sleep(4);
    }

    // Primary VNIC
    $atts = $api->call($config, "$iaas/vnicAttachments/", 'GET', null, [
        'compartmentId' => $tenancy, 'instanceId' => $instanceId,
    ]);
    $vnicId = null;
    foreach ($atts as $a) {
        if (($a['lifecycleState'] ?? '') === 'ATTACHED') { $vnicId = $a['vnicId']; break; }
    }
    if (!$vnicId) { echo "$name: no attached VNIC yet, skipping\n"; continue; }

    // Primary private IP
    $privs = $api->call($config, "$iaas/privateIps/", 'GET', null, ['vnicId' => $vnicId]);
    $privId = null;
    foreach ($privs as $p) {
        if (($p['isPrimary'] ?? false) === true) { $privId = $p['id']; break; }
    }
    if (!$privId && !empty($privs)) { $privId = $privs[0]['id']; }
    if (!$privId) { echo "$name: no private IP found, skipping\n"; continue; }

    // Assign ephemeral public IP
    try {
        $body = json_encode([
            'compartmentId' => $tenancy,
            'lifetime' => 'EPHEMERAL',
            'privateIpId' => $privId,
        ]);
        $pub = $api->call($config, "$iaas/publicIps/", 'POST', $body);
        echo "$name: PUBLIC_IP=" . ($pub['ipAddress'] ?? '?') . "\n";
    } catch (Throwable $e) {
        echo "$name: public IP assign failed -> " . $e->getMessage() . "\n";
    }
}
