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

$region = getenv('OCI_REGION');
$tenancy = getenv('OCI_TENANCY_ID');
$iaas = "https://iaas.$region.oraclecloud.com/20160918";
$api = new OciApi();

function logln(string $m): void { echo $m . "\n"; }

// Reuse existing VCN if present
$vcns = $api->call($config, "$iaas/vcns/", 'GET', null, ['compartmentId' => $tenancy]);
if (!empty($vcns)) {
    logln('VCN already exists: ' . $vcns[0]['id'] . ' - aborting to avoid duplicates.');
    $subnets = $api->call($config, "$iaas/subnets/", 'GET', null, ['compartmentId' => $tenancy]);
    if (!empty($subnets)) {
        logln('SUBNET_ID=' . $subnets[0]['id']);
    }
    return;
}

// 1) Create VCN
logln('Creating VCN...');
$vcnBody = json_encode([
    'compartmentId' => $tenancy,
    'cidrBlocks' => ['10.0.0.0/16'],
    'displayName' => 'free-vcn',
    'dnsLabel' => 'freevcn',
]);
$vcn = $api->call($config, "$iaas/vcns/", 'POST', $vcnBody);
$vcnId = $vcn['id'];
$routeTableId = $vcn['defaultRouteTableId'];
$securityListId = $vcn['defaultSecurityListId'];
$dhcpId = $vcn['defaultDhcpOptionsId'];
logln("VCN: $vcnId");

// Poll until available
for ($i = 0; $i < 20; $i++) {
    $v = $api->call($config, "$iaas/vcns/$vcnId", 'GET');
    if (($v['lifecycleState'] ?? '') === 'AVAILABLE') { break; }
    sleep(2);
}

// 2) Internet Gateway
logln('Creating Internet Gateway...');
$igBody = json_encode([
    'compartmentId' => $tenancy,
    'vcnId' => $vcnId,
    'isEnabled' => true,
    'displayName' => 'free-ig',
]);
$ig = $api->call($config, "$iaas/internetGateways/", 'POST', $igBody);
$igId = $ig['id'];
logln("IG: $igId");

// 3) Route rule 0.0.0.0/0 -> IG on default route table
logln('Adding default route to Internet Gateway...');
$rtBody = json_encode([
    'routeRules' => [[
        'destination' => '0.0.0.0/0',
        'destinationType' => 'CIDR_BLOCK',
        'networkEntityId' => $igId,
    ]],
]);
$api->call($config, "$iaas/routeTables/$routeTableId", 'PUT', $rtBody);

// 4) Ensure SSH ingress on default security list (idempotent merge)
logln('Ensuring SSH ingress rule...');
$sl = $api->call($config, "$iaas/securityLists/$securityListId", 'GET');
$ingress = $sl['ingressSecurityRules'] ?? [];
$hasSsh = false;
foreach ($ingress as $r) {
    if (($r['protocol'] ?? '') === '6' && isset($r['tcpOptions']['destinationPortRange'])
        && ($r['tcpOptions']['destinationPortRange']['min'] ?? null) <= 22
        && ($r['tcpOptions']['destinationPortRange']['max'] ?? null) >= 22) {
        $hasSsh = true; break;
    }
}
if (!$hasSsh) {
    $ingress[] = [
        'protocol' => '6',
        'source' => '0.0.0.0/0',
        'isStateless' => false,
        'tcpOptions' => ['destinationPortRange' => ['min' => 22, 'max' => 22]],
    ];
    $slBody = json_encode([
        'ingressSecurityRules' => $ingress,
        'egressSecurityRules' => $sl['egressSecurityRules'] ?? [],
    ]);
    $api->call($config, "$iaas/securityLists/$securityListId", 'PUT', $slBody);
    logln('SSH ingress added.');
} else {
    logln('SSH ingress already present.');
}

// 5) Create public regional subnet
logln('Creating public subnet...');
$subnetBody = json_encode([
    'compartmentId' => $tenancy,
    'vcnId' => $vcnId,
    'cidrBlock' => '10.0.1.0/24',
    'displayName' => 'free-subnet',
    'dnsLabel' => 'freesubnet',
    'prohibitPublicIpOnVnic' => false,
    'routeTableId' => $routeTableId,
    'securityListIds' => [$securityListId],
    'dhcpOptionsId' => $dhcpId,
]);
$subnet = $api->call($config, "$iaas/subnets/", 'POST', $subnetBody);
logln('SUBNET_ID=' . $subnet['id']);
logln('DONE');
