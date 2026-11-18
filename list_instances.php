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
foreach ($instances as $i) {
    if (($i['lifecycleState'] ?? '') === 'TERMINATED') continue;
    $publicIp = '-';
    try {
        $atts = $api->call($config, "$iaas/vnicAttachments/", 'GET', null, [
            'compartmentId' => $tenancy, 'instanceId' => $i['id'],
        ]);
        foreach ($atts as $a) {
            if (($a['lifecycleState'] ?? '') === 'ATTACHED') {
                $vnic = $api->call($config, "$iaas/vnics/{$a['vnicId']}", 'GET');
                $publicIp = $vnic['publicIp'] ?? '-';
                break;
            }
        }
    } catch (Throwable $e) {}
    printf("%-22s %-22s %-10s %s/%-2s %-26s IP=%s\n",
        $i['displayName'] ?? '?',
        $i['shape'] ?? '?',
        $i['lifecycleState'] ?? '?',
        $i['shapeConfig']['ocpus'] ?? '?',
        $i['shapeConfig']['memoryInGBs'] ?? '?',
        $i['availabilityDomain'] ?? '?',
        $publicIp
    );
}
