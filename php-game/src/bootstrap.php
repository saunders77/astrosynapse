<?php
declare(strict_types=1);
require_once __DIR__.'/Engine.php';
require_once __DIR__.'/Encoder.php';
require_once __DIR__.'/Actor.php';
require_once __DIR__.'/Lethal.php';
require_once __DIR__.'/Models.php';
require_once __DIR__.'/Session.php';
function astro_boot(): array {
    if(PHP_VERSION_ID<80100 || PHP_INT_SIZE<8) throw new \RuntimeException('64-bit PHP 8.1 or newer is required');
    $config=require __DIR__.'/../config.php';
    if(is_file(__DIR__.'/../config.local.php')) $config=array_replace($config,require __DIR__.'/../config.local.php');
    ini_set('display_errors','0');
    ini_set('session.use_strict_mode','1');
    if($config['session_directory']) session_save_path($config['session_directory']);
    session_name('astrosynapse_game');
    $path=str_replace('\\','/',dirname($_SERVER['SCRIPT_NAME']??'/'));
    session_set_cookie_params(['httponly'=>true,'samesite'=>'Strict','secure'=>!empty($_SERVER['HTTPS']) && $_SERVER['HTTPS']!=='off','path'=>rtrim($path,'/').'/']);
    if(!session_start()) throw new \RuntimeException('PHP session storage is unavailable. Configure a writable session_directory.');
    $_SESSION['csrf']??=bin2hex(random_bytes(32));
    header('Cache-Control: no-store');
    header('X-Content-Type-Options: nosniff');
    header("Content-Security-Policy: default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'");
    return $config;
}
