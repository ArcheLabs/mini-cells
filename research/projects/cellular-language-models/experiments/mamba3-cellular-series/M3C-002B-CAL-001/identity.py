"""Bind a real saved Kaggle Script Version ID to the current runtime and source."""
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid
from urllib.parse import urlparse,parse_qs


def version_ids(value):
    result=set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key.lower().replace('_','')=='scriptversionid' and str(item).isdigit() and int(item)>0:result.add(str(item))
            elif isinstance(item,(dict,list)):result.update(version_ids(item))
    elif isinstance(value,list):
        for item in value:result.update(version_ids(item))
    return result


def validate(runtime_ids,saved_id,runtime_source,saved_source,tag):
    if len(runtime_ids)!=1:raise RuntimeError('CURRENT_SCRIPT_VERSION_ID_UNAVAILABLE_OR_AMBIGUOUS: use Save Version / Save & Run All; do not run an unsaved interactive draft')
    current=next(iter(runtime_ids))
    if saved_id is not None and current!=str(saved_id):raise RuntimeError('CURRENT_SCRIPT_VERSION_ID_DIFFERS_FROM_SAVED_API_VERSION; no latest-version guessing')
    if tag not in runtime_source or tag not in saved_source:raise RuntimeError('Saved/runtime notebook does not contain the frozen calibration protocol tag')
    return current


def pending(kernel,tag,runtime_source,reason,allowed):
    if not allowed:raise RuntimeError(reason)
    if tag not in runtime_source:raise RuntimeError('Current notebook does not contain the frozen calibration tag')
    return {'status':'ID_PENDING','kernel':kernel,'script_version_id':None,
            'session_id':uuid.uuid4().hex,'protocol_tag':tag,
            'runtime_source_sha256':hashlib.sha256(runtime_source.encode()).hexdigest(),
            'run_type':os.environ.get('KAGGLE_KERNEL_RUN_TYPE'),'reason':reason,
            'scope':'User-authorized development calibration only; saved-version identity must be supplemented later',
            'formal_training_allowed':False}


def saved_version_record(url,decision):
    parsed=urlparse(url);query=parse_qs(parsed.query)
    kernel=(decision.get('execution') or {}).get('kernel')
    allowed_paths=('/code/'+str(kernel),'/'+str(kernel))
    values=query.get('scriptVersionId',[])
    if parsed.scheme!='https' or parsed.hostname not in ('www.kaggle.com','kaggle.com') or parsed.path.rstrip('/') not in allowed_paths:
        raise ValueError('URL must identify the saved version of this Kaggle notebook')
    if len(values)!=1 or not values[0].isdigit() or int(values[0])<=0:
        raise ValueError('A real scriptVersionId query parameter is required; version ordinals are not accepted')
    known=(decision.get('execution') or {}).get('script_version_id')
    if known is not None and str(known)!=values[0]:raise ValueError('Saved URL conflicts with the already recorded runtime Script Version ID')
    return {'status':'SAVED_VERSION_URL_RECORDED','run_id':decision['run_id'],
            'source_commit':decision['source_commit'],'kernel':kernel,'script_version_id':values[0],
            'saved_version_url':f'https://www.kaggle.com/code/{kernel}?scriptVersionId={values[0]}','provenance':'USER_SUPPLIED_PLATFORM_SAVED_VERSION_URL',
            'independent_platform_verification':'NOT_PERFORMED',
            'note':'Append-only supplement; original execution report and pending identity are preserved'}


def capture(root,kernel,tag,allow_deferred=False):
    ids=set();sources=[];runtime_source=''
    value=os.environ.get('KAGGLE_SCRIPT_VERSION_ID')
    if value and value.isdigit():ids.add(value);sources.append('runtime KAGGLE_SCRIPT_VERSION_ID')
    # Never print/store the token or unrestricted claims.
    token=os.environ.get('KAGGLE_USER_SECRETS_TOKEN','')
    try:
        payload=token.split('.')[1];claims=json.loads(base64.urlsafe_b64decode(payload+'='*((-len(payload))%4)))
        found=version_ids(claims);ids.update(found)
        if found:sources.append('current-runtime credential scriptVersionId claim; cross-checked with Kaggle API')
    except (ValueError,IndexError):pass
    try:
        from kaggle_session import UserSessionClient
        session=UserSessionClient();current=session.web_client.make_post_request({'UseDraft':False},session.GET_SOURCE_ENDPOINT,timeout=20)
        runtime_source=json.dumps(current)
        found=version_ids(current);ids.update(found)
        if found:sources.append('current-session GetKernelRunSourceForCaipRequest UseDraft=False')
    except Exception:pass
    script=Path('/kaggle/src/script.py')
    if script.exists() and tag in script.read_text():runtime_source=script.read_text()
    if not ids:return pending(kernel,tag,runtime_source,'CURRENT_SCRIPT_VERSION_ID_NOT_EXPOSED_BY_RUNTIME',allow_deferred)
    if len(ids)!=1:raise RuntimeError('Ambiguous current Script Version IDs; conflicting IDs cannot be deferred')
    # Reuse the repository's authenticated, source-pinned Kaggle client setup.
    sys.path.insert(0,str(root/'research/projects/cellular-language-models/experiments/epclm-002-a0-persistent-state-recall/scripts'))
    from kaggle_output_recovery import _load_backend
    try:
        backend,_=_load_backend(root)
        from kaggle.api.kaggle_api_extended import ApiGetKernelRequest
        request=ApiGetKernelRequest();request.user_name,request.kernel_slug=kernel.split('/')
        with backend.api.build_kaggle_client() as client:response=client.kernels.kernels_api_client.get_kernel(request)
    except Exception as error:
        return pending(kernel,tag,runtime_source,'KAGGLE_IDENTITY_API_UNAVAILABLE:'+type(error).__name__,allow_deferred)
    saved_id=getattr(response.metadata,'script_version_id',None)
    if saved_id is None:saved_id=getattr(response.blob,'script_version_id',None)
    if not str(saved_id).isdigit() or int(saved_id)<=0:saved_id=None
    saved_source=response.blob.source
    if not runtime_source:
        # Runtime ID must still be explicit; source alone never supplies a guessed ID.
        try:
            from kaggle_session import UserSessionClient
            runtime_source=json.dumps(UserSessionClient().get_exportable_ipynb())
        except Exception:pass
    # The public SDK may omit scriptVersionId entirely. An ID explicitly exposed by
    # the current session is still usable; an API ordinal/kernel ID is never substituted.
    if not sources or (sources==['runtime KAGGLE_SCRIPT_VERSION_ID'] and saved_id is None):
        return pending(kernel,tag,runtime_source,'PLATFORM_VERSION_ID_NOT_CORROBORATED',allow_deferred)
    current=validate(ids,saved_id,runtime_source,saved_source,tag)
    return {'status':'VERIFIED_CURRENT_RUNTIME_AND_SAVED_SOURCE','session_id':uuid.uuid4().hex,'kernel':kernel,'script_version_id':current,
            'version_id_sources':sources,'api_script_version_id_exposed':saved_id is not None,
            'version_ordinal':getattr(response.metadata,'current_version_number',None),'saved_source_sha256':hashlib.sha256(saved_source.encode()).hexdigest(),
            'protocol_tag':tag,'run_type':os.environ.get('KAGGLE_KERNEL_RUN_TYPE'),
            'note':'No ordinal/time/Git-SHA or unbound latest-version substitution'}
