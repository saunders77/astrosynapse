<?php
declare(strict_types=1);
require __DIR__.'/src/bootstrap.php';
use Astro\Models;
use Astro\Session;
header('Content-Type: application/json; charset=utf-8');
try {
    $config=astro_boot();
    $method=$_SERVER['REQUEST_METHOD']??'GET';
    if($method==='GET') {
        $data=['csrf'=>$_SESSION['csrf'],'models'=>Models::publicList(),'admin_enabled'=>$config['admin_password_hash']!=='','admin'=>($_SESSION['admin']??false)===true,'game'=>null];
        if(isset($_SESSION['game'])) $data['game']=Session::advance($_SESSION['game']);
    } elseif($method==='POST') {
        if(!hash_equals($_SESSION['csrf'],$_SERVER['HTTP_X_CSRF_TOKEN']??'')) { http_response_code(403); throw new InvalidArgumentException('Refresh this page before continuing'); }
        $type=$_SERVER['CONTENT_TYPE']??'';
        if(strncmp($type,'multipart/form-data',19)===0) $in=$_POST;
        else { if((int)($_SERVER['CONTENT_LENGTH']??0)>8192) throw new InvalidArgumentException('Request too large'); $in=json_decode(file_get_contents('php://input'),true,32,JSON_THROW_ON_ERROR); }
        if(!is_array($in)) throw new InvalidArgumentException('Invalid request');
        $op=$in['op']??'';
        if($op==='new') { $next=Session::start((string)($in['model']??''),($in['starts']??true)===true); $data=['game'=>Session::advance($next)]; $_SESSION['game']=$next; }
        elseif(in_array($op,['choose','play_all','advance'],true)) {
            if(!isset($_SESSION['game']) || ($in['id']??null)!==$_SESSION['game']['id'] || ($in['revision']??null)!==$_SESSION['game']['revision']) { http_response_code(409); throw new InvalidArgumentException('The game changed in another request. Refresh to continue.'); }
            if($op==='choose' && !is_int($in['action_id']??null)) throw new InvalidArgumentException('Choose a legal move');
            // Commit only after successful execution; stale or invalid actions
            // cannot partially mutate a game, even during a Play all batch.
            $next=$_SESSION['game']; $data=['game'=>Session::advance($next,$op,$in['action_id']??null)]; $_SESSION['game']=$next;
        } elseif($op==='login') {
            if(!$config['admin_password_hash']) throw new InvalidArgumentException('Set admin_password_hash in config.php to enable model management');
            if(time()<($_SESSION['login_after']??0)) throw new InvalidArgumentException('Please wait before trying again');
            $_SESSION['login_after']=time()+3;
            if(!password_verify((string)($in['password']??''),$config['admin_password_hash'])) { http_response_code(403); throw new InvalidArgumentException('Incorrect password'); }
            session_regenerate_id(true); $_SESSION['admin']=true; $data=['admin'=>true];
        } elseif($op==='logout') { unset($_SESSION['admin']); $data=['admin'=>false]; }
        elseif(in_array($op,['rename','upload'],true)) {
            if(($_SESSION['admin']??false)!==true) { http_response_code(403); throw new InvalidArgumentException('Sign in to manage models'); }
            if($op==='rename') {
                $id=(string)($in['model']??''); $name=Models::name((string)($in['name']??'')); Models::get($id);
                Models::update(function($rows) use($id,$name) { foreach($rows as &$r) if($r['id']===$id) $r['name']=$name; return $rows; });
            } else {
                $file=$_FILES['model_file']??null;
                if(!$file || $file['error']!==UPLOAD_ERR_OK || !is_uploaded_file($file['tmp_name'])) throw new InvalidArgumentException('Upload failed. Check upload_max_filesize and post_max_size on your server.');
                Models::import($file['tmp_name'],(string)($in['name']??''));
            }
            $data=['models'=>Models::publicList()];
        } else throw new InvalidArgumentException('Unknown request');
    } else { http_response_code(405); throw new InvalidArgumentException('Method not allowed'); }
    echo json_encode($data,JSON_THROW_ON_ERROR|JSON_UNESCAPED_UNICODE);
} catch(InvalidArgumentException|JsonException $e) { if(http_response_code()<400) http_response_code(400); echo json_encode(['error'=>$e->getMessage()]); }
catch(Throwable $e) { error_log('Astrosynapse: '.$e); http_response_code(500); echo json_encode(['error'=>'The server could not complete this move. Your last saved game is intact. Check the PHP error log and hosting requirements.']); }
