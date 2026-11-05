"""Explicit requested/selected policy; framework compatibility is not native hardware."""
import torch

def select(requested='auto',device=None,diagnostic_bf16=False):
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cuda=device.type=='cuda';cap=tuple(torch.cuda.get_device_capability(device)) if cuda else (0,0)
    framework=bool(cuda and torch.cuda.is_bf16_supported());native=framework and cap[0]>=8
    fp16=bool(cuda and cap>=(5,3));supported=('auto','fp32','fp16','bf16')
    if requested not in supported:raise ValueError('unknown precision '+requested)
    chosen=('bf16' if native else 'fp16' if fp16 else 'fp32') if requested=='auto' else requested
    if chosen=='bf16' and not native and not (diagnostic_bf16 and framework):raise ValueError('BF16 is not native on this GPU; diagnostic opt-in required')
    if chosen=='fp16' and not fp16:raise ValueError('FP16 unsupported on selected device')
    return {'requested_precision':requested,'selected_precision':chosen,'bf16_framework_supported':framework,'bf16_native_hardware_supported':native,'fp16_hardware_supported':fp16,'scaler_enabled':chosen=='fp16','gpu_capability':list(cap),'gpu_name':torch.cuda.get_device_name(device) if cuda else 'CPU','torch_version':str(torch.__version__),'cuda_version':torch.version.cuda,'diagnostic_non_native_bf16':chosen=='bf16' and not native,'selection_reason':'auto chooses native BF16 on CC>=8, otherwise FP16+GradScaler when supported; never auto-emulated BF16' if requested=='auto' else 'explicit request; non-native BF16 diagnostic only'}

def autocast(policy,device):return torch.autocast(device_type=torch.device(device).type,dtype=torch.float16 if policy['selected_precision']=='fp16' else torch.bfloat16,enabled=policy['selected_precision']!='fp32')
