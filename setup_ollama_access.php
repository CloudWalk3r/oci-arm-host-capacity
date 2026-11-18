<?php
declare(strict_types=1);

error_reporting(E_ALL & ~E_DEPRECATED);
require __DIR__ . '/vendor/autoload.php';

use Dotenv\Dotenv;
use Hitrov\OciApi;
use Hitrov\OciConfig;

$dotenv = Dotenv::createUnsafeImmutable(__DIR__, '.env');
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

// --- Private + public IPs per instance ---
echo "===== INSTANCE IPs =====\n";
$instances = $api->getInstances($config);
foreach ($instances as $i) {
    if (($i['lifecycleState'] ?? '') === 'TERMINATED') continue;
    $priv = '-'; $pub = '-';
    $atts = $api->call($config, "$iaas/vnicAttachments/", 'GET', null, [
        'compartmentId' => $tenancy, 'instanceId' => $i['id'],
    ]);
    foreach ($atts as $a) {
        if (($a['lifecycleState'] ?? '') === 'ATTACHED') {
            $vnic = $api->call($config, "$iaas/vnics/{$a['vnicId']}", 'GET');
            $priv = $vnic['privateIp'] ?? '-';
            $pub = $vnic['publicIp'] ?? '-';
            break;
        }
    }
    printf("%-22s %-22s priv=%-12s pub=%s\n",
        $i['displayName'] ?? '?', $i['shape'] ?? '?', $priv, $pub);
}

// --- Add Ollama ingress (TCP 11434 from VCN CIDR) ---
echo "\n===== SECURITY LIST =====\n";
$vcns = $api->call($config, "$iaas/vcns/", 'GET', null, ['compartmentId' => $tenancy]);
if (empty($vcns)) { echo "No VCN found\n"; return; }
$vcn = $vcns[0];
$vcnCidr = $vcn['cidrBlocks'][0] ?? '10.0.0.0/16';
$slId = $vcn['defaultSecurityListId'];

$sl = $api->call($config, "$iaas/securityLists/$slId", 'GET');
$ingress = $sl['ingressSecurityRules'] ?? [];
$has = false;
foreach ($ingress as $r) {
    if (($r['protocol'] ?? '') === '6'
        && ($r['tcpOptions']['destinationPortRange']['min'] ?? null) <= 11434
        && ($r['tcpOptions']['destinationPortRange']['max'] ?? null) >= 11434) {
        $has = true; break;
    }
}
if ($has) {
    echo "Ollama ingress (11434) already present.\n";
} else {
    $ingress[] = [
        'protocol' => '6',
        'source' => $vcnCidr,
        'isStateless' => false,
        'description' => 'Ollama API from within VCN',
        'tcpOptions' => ['destinationPortRange' => ['min' => 11434, 'max' => 11434]],
    ];
    $body = json_encode([
        'ingressSecurityRules' => $ingress,
        'egressSecurityRules' => $sl['egressSecurityRules'] ?? [],
    ]);
    $api->call($config, "$iaas/securityLists/$slId", 'PUT', $body);
    echo "Added ingress TCP 11434 from $vcnCidr\n";
}
