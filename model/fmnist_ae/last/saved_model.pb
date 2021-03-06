??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.5.0-rc02v1.12.1-53831-ga8b6d5ff93a8??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: *
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
: *
dtype0
?
conv2d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_9/kernel
{
#conv2d_9/kernel/Read/ReadVariableOpReadVariableOpconv2d_9/kernel*&
_output_shapes
: *
dtype0
r
conv2d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_9/bias
k
!conv2d_9/bias/Read/ReadVariableOpReadVariableOpconv2d_9/bias*
_output_shapes
:*
dtype0
?
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
_output_shapes
:*
dtype0
?
conv2d_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_11/kernel
}
$conv2d_11/kernel/Read/ReadVariableOpReadVariableOpconv2d_11/kernel*&
_output_shapes
:*
dtype0
t
conv2d_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_11/bias
m
"conv2d_11/bias/Read/ReadVariableOpReadVariableOpconv2d_11/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_12/kernel
?
.conv2d_transpose_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_12/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_12/bias
?
,conv2d_transpose_12/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_12/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameconv2d_transpose_13/kernel
?
.conv2d_transpose_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_13/kernel*&
_output_shapes
:*
dtype0
?
conv2d_transpose_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_13/bias
?
,conv2d_transpose_13/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_13/bias*
_output_shapes
:*
dtype0
?
conv2d_transpose_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_14/kernel
?
.conv2d_transpose_14/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_14/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_nameconv2d_transpose_14/bias
?
,conv2d_transpose_14/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_14/bias*
_output_shapes
: *
dtype0
?
conv2d_transpose_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameconv2d_transpose_15/kernel
?
.conv2d_transpose_15/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/kernel*&
_output_shapes
: *
dtype0
?
conv2d_transpose_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameconv2d_transpose_15/bias
?
,conv2d_transpose_15/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_15/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_8/kernel/m
?
*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_8/bias/m
y
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2d_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_9/kernel/m
?
*Adam/conv2d_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_9/bias/m
y
(Adam/conv2d_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_10/kernel/m
?
+Adam/conv2d_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_10/bias/m
{
)Adam/conv2d_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_11/kernel/m
?
+Adam/conv2d_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_11/bias/m
{
)Adam/conv2d_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_12/kernel/m
?
5Adam/conv2d_transpose_12/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_12/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_12/bias/m
?
3Adam/conv2d_transpose_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_12/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_13/kernel/m
?
5Adam/conv2d_transpose_13/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_13/kernel/m*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_13/bias/m
?
3Adam/conv2d_transpose_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_13/bias/m*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_14/kernel/m
?
5Adam/conv2d_transpose_14/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_14/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_14/bias/m
?
3Adam/conv2d_transpose_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_14/bias/m*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_15/kernel/m
?
5Adam/conv2d_transpose_15/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_15/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_15/bias/m
?
3Adam/conv2d_transpose_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_15/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_8/kernel/v
?
*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nameAdam/conv2d_8/bias/v
y
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2d_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameAdam/conv2d_9/kernel/v
?
*Adam/conv2d_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_9/bias/v
y
(Adam/conv2d_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_9/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_10/kernel/v
?
+Adam/conv2d_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_10/bias/v
{
)Adam/conv2d_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_10/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_11/kernel/v
?
+Adam/conv2d_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_11/bias/v
{
)Adam/conv2d_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_11/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_12/kernel/v
?
5Adam/conv2d_transpose_12/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_12/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_12/bias/v
?
3Adam/conv2d_transpose_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_12/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/conv2d_transpose_13/kernel/v
?
5Adam/conv2d_transpose_13/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_13/kernel/v*&
_output_shapes
:*
dtype0
?
Adam/conv2d_transpose_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_13/bias/v
?
3Adam/conv2d_transpose_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_13/bias/v*
_output_shapes
:*
dtype0
?
!Adam/conv2d_transpose_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_14/kernel/v
?
5Adam/conv2d_transpose_14/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_14/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!Adam/conv2d_transpose_14/bias/v
?
3Adam/conv2d_transpose_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_14/bias/v*
_output_shapes
: *
dtype0
?
!Adam/conv2d_transpose_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!Adam/conv2d_transpose_15/kernel/v
?
5Adam/conv2d_transpose_15/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/conv2d_transpose_15/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv2d_transpose_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/conv2d_transpose_15/bias/v
?
3Adam/conv2d_transpose_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_15/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?a
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?a
value?aB?a B?`
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
 
?
layer-0

layer_with_weights-0

layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	variables
regularization_losses
trainable_variables
	keras_api
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	variables
regularization_losses
trainable_variables
 	keras_api
?
!iter

"beta_1

#beta_2
	$decay
%learning_rate&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?
v
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
414
515
 
v
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
414
515
?
6metrics

7layers
	variables
8non_trainable_variables
9layer_metrics
regularization_losses
trainable_variables
:layer_regularization_losses
 
h

&kernel
'bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
R
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
h

(kernel
)bias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
R
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
h

*kernel
+bias
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
h

,kernel
-bias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
R
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
8
&0
'1
(2
)3
*4
+5
,6
-7
 
8
&0
'1
(2
)3
*4
+5
,6
-7
?
Wmetrics

Xlayers
	variables
Ynon_trainable_variables
Zlayer_metrics
regularization_losses
trainable_variables
[layer_regularization_losses
 
R
\	variables
]regularization_losses
^trainable_variables
_	keras_api
h

.kernel
/bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
h

0kernel
1bias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
R
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
h

2kernel
3bias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
R
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
h

4kernel
5bias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
8
.0
/1
02
13
24
35
46
57
 
8
.0
/1
02
13
24
35
46
57
?
xmetrics

ylayers
	variables
znon_trainable_variables
{layer_metrics
regularization_losses
trainable_variables
|layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_8/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_8/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEconv2d_9/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_9/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_10/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_10/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEconv2d_11/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEconv2d_11/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_transpose_12/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d_transpose_12/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_transpose_13/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_transpose_13/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_transpose_14/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_transpose_14/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_transpose_15/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_transpose_15/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE

}0

0
1
2
 
 
 

&0
'1
 

&0
'1
?
~metrics

layers
;	variables
?non_trainable_variables
?layer_metrics
<regularization_losses
=trainable_variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?layers
?	variables
?non_trainable_variables
?layer_metrics
@regularization_losses
Atrainable_variables
 ?layer_regularization_losses

(0
)1
 

(0
)1
?
?metrics
?layers
C	variables
?non_trainable_variables
?layer_metrics
Dregularization_losses
Etrainable_variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?layers
G	variables
?non_trainable_variables
?layer_metrics
Hregularization_losses
Itrainable_variables
 ?layer_regularization_losses

*0
+1
 

*0
+1
?
?metrics
?layers
K	variables
?non_trainable_variables
?layer_metrics
Lregularization_losses
Mtrainable_variables
 ?layer_regularization_losses

,0
-1
 

,0
-1
?
?metrics
?layers
O	variables
?non_trainable_variables
?layer_metrics
Pregularization_losses
Qtrainable_variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?layers
S	variables
?non_trainable_variables
?layer_metrics
Tregularization_losses
Utrainable_variables
 ?layer_regularization_losses
 
8
0

1
2
3
4
5
6
7
 
 
 
 
 
 
?
?metrics
?layers
\	variables
?non_trainable_variables
?layer_metrics
]regularization_losses
^trainable_variables
 ?layer_regularization_losses

.0
/1
 

.0
/1
?
?metrics
?layers
`	variables
?non_trainable_variables
?layer_metrics
aregularization_losses
btrainable_variables
 ?layer_regularization_losses

00
11
 

00
11
?
?metrics
?layers
d	variables
?non_trainable_variables
?layer_metrics
eregularization_losses
ftrainable_variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?layers
h	variables
?non_trainable_variables
?layer_metrics
iregularization_losses
jtrainable_variables
 ?layer_regularization_losses

20
31
 

20
31
?
?metrics
?layers
l	variables
?non_trainable_variables
?layer_metrics
mregularization_losses
ntrainable_variables
 ?layer_regularization_losses
 
 
 
?
?metrics
?layers
p	variables
?non_trainable_variables
?layer_metrics
qregularization_losses
rtrainable_variables
 ?layer_regularization_losses

40
51
 

40
51
?
?metrics
?layers
t	variables
?non_trainable_variables
?layer_metrics
uregularization_losses
vtrainable_variables
 ?layer_regularization_losses
 
8
0
1
2
3
4
5
6
7
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
nl
VARIABLE_VALUEAdam/conv2d_8/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_8/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_9/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_9/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_10/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_10/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_11/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_11/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/conv2d_transpose_12/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_transpose_12/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/conv2d_transpose_13/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_transpose_13/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/conv2d_transpose_14/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_transpose_14/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/conv2d_transpose_15/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_transpose_15/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_8/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_8/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/conv2d_9/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d_9/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_10/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_10/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/conv2d_11/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/conv2d_11/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/conv2d_transpose_12/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d_transpose_12/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/conv2d_transpose_13/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_transpose_13/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/conv2d_transpose_14/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_transpose_14/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/conv2d_transpose_15/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_transpose_15/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_5Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5conv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_transpose_12/kernelconv2d_transpose_12/biasconv2d_transpose_13/kernelconv2d_transpose_13/biasconv2d_transpose_14/kernelconv2d_transpose_14/biasconv2d_transpose_15/kernelconv2d_transpose_15/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_21040
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp#conv2d_9/kernel/Read/ReadVariableOp!conv2d_9/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp$conv2d_11/kernel/Read/ReadVariableOp"conv2d_11/bias/Read/ReadVariableOp.conv2d_transpose_12/kernel/Read/ReadVariableOp,conv2d_transpose_12/bias/Read/ReadVariableOp.conv2d_transpose_13/kernel/Read/ReadVariableOp,conv2d_transpose_13/bias/Read/ReadVariableOp.conv2d_transpose_14/kernel/Read/ReadVariableOp,conv2d_transpose_14/bias/Read/ReadVariableOp.conv2d_transpose_15/kernel/Read/ReadVariableOp,conv2d_transpose_15/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp*Adam/conv2d_9/kernel/m/Read/ReadVariableOp(Adam/conv2d_9/bias/m/Read/ReadVariableOp+Adam/conv2d_10/kernel/m/Read/ReadVariableOp)Adam/conv2d_10/bias/m/Read/ReadVariableOp+Adam/conv2d_11/kernel/m/Read/ReadVariableOp)Adam/conv2d_11/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_12/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_12/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_13/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_13/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_14/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_14/bias/m/Read/ReadVariableOp5Adam/conv2d_transpose_15/kernel/m/Read/ReadVariableOp3Adam/conv2d_transpose_15/bias/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOp*Adam/conv2d_9/kernel/v/Read/ReadVariableOp(Adam/conv2d_9/bias/v/Read/ReadVariableOp+Adam/conv2d_10/kernel/v/Read/ReadVariableOp)Adam/conv2d_10/bias/v/Read/ReadVariableOp+Adam/conv2d_11/kernel/v/Read/ReadVariableOp)Adam/conv2d_11/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_12/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_12/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_13/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_13/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_14/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_14/bias/v/Read/ReadVariableOp5Adam/conv2d_transpose_15/kernel/v/Read/ReadVariableOp3Adam/conv2d_transpose_15/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_22040
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d_8/kernelconv2d_8/biasconv2d_9/kernelconv2d_9/biasconv2d_10/kernelconv2d_10/biasconv2d_11/kernelconv2d_11/biasconv2d_transpose_12/kernelconv2d_transpose_12/biasconv2d_transpose_13/kernelconv2d_transpose_13/biasconv2d_transpose_14/kernelconv2d_transpose_14/biasconv2d_transpose_15/kernelconv2d_transpose_15/biastotalcountAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/conv2d_9/kernel/mAdam/conv2d_9/bias/mAdam/conv2d_10/kernel/mAdam/conv2d_10/bias/mAdam/conv2d_11/kernel/mAdam/conv2d_11/bias/m!Adam/conv2d_transpose_12/kernel/mAdam/conv2d_transpose_12/bias/m!Adam/conv2d_transpose_13/kernel/mAdam/conv2d_transpose_13/bias/m!Adam/conv2d_transpose_14/kernel/mAdam/conv2d_transpose_14/bias/m!Adam/conv2d_transpose_15/kernel/mAdam/conv2d_transpose_15/bias/mAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/conv2d_9/kernel/vAdam/conv2d_9/bias/vAdam/conv2d_10/kernel/vAdam/conv2d_10/bias/vAdam/conv2d_11/kernel/vAdam/conv2d_11/bias/v!Adam/conv2d_transpose_12/kernel/vAdam/conv2d_transpose_12/bias/v!Adam/conv2d_transpose_13/kernel/vAdam/conv2d_transpose_13/bias/v!Adam/conv2d_transpose_14/kernel/vAdam/conv2d_transpose_14/bias/v!Adam/conv2d_transpose_15/kernel/vAdam/conv2d_transpose_15/bias/v*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_22215??
?
K
/__inference_up_sampling2d_6_layer_call_fn_20369

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_203632
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_19947

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_19959

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_20499

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
(__inference_model_10_layer_call_fn_20070
input_5!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_200512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?&
?
C__inference_model_11_layer_call_and_return_conditional_losses_20693
input_63
conv2d_transpose_12_20670:'
conv2d_transpose_12_20672:3
conv2d_transpose_13_20675:'
conv2d_transpose_13_20677:3
conv2d_transpose_14_20681: '
conv2d_transpose_14_20683: 3
conv2d_transpose_15_20687: '
conv2d_transpose_15_20689:
identity??+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?
reshape_3/PartitionedCallPartitionedCallinput_6*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204992
reshape_3/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_12_20670conv2d_transpose_12_20672*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_202952-
+conv2d_transpose_12/StatefulPartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_20675conv2d_transpose_13_20677*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_203402-
+conv2d_transpose_13/StatefulPartitionedCall?
up_sampling2d_6/PartitionedCallPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_203632!
up_sampling2d_6/PartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_14_20681conv2d_transpose_14_20683*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_204042-
+conv2d_transpose_14/StatefulPartitionedCall?
up_sampling2d_7/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_204272!
up_sampling2d_7/PartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0conv2d_transpose_15_20687conv2d_transpose_15_20689*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_204682-
+conv2d_transpose_15/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_15/StatefulPartitionedCall:output:0,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_6
?
?
(__inference_model_12_layer_call_fn_21114

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_208472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_19941
input_5S
9model_12_model_10_conv2d_8_conv2d_readvariableop_resource: H
:model_12_model_10_conv2d_8_biasadd_readvariableop_resource: S
9model_12_model_10_conv2d_9_conv2d_readvariableop_resource: H
:model_12_model_10_conv2d_9_biasadd_readvariableop_resource:T
:model_12_model_10_conv2d_10_conv2d_readvariableop_resource:I
;model_12_model_10_conv2d_10_biasadd_readvariableop_resource:T
:model_12_model_10_conv2d_11_conv2d_readvariableop_resource:I
;model_12_model_10_conv2d_11_biasadd_readvariableop_resource:h
Nmodel_12_model_11_conv2d_transpose_12_conv2d_transpose_readvariableop_resource:S
Emodel_12_model_11_conv2d_transpose_12_biasadd_readvariableop_resource:h
Nmodel_12_model_11_conv2d_transpose_13_conv2d_transpose_readvariableop_resource:S
Emodel_12_model_11_conv2d_transpose_13_biasadd_readvariableop_resource:h
Nmodel_12_model_11_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: S
Emodel_12_model_11_conv2d_transpose_14_biasadd_readvariableop_resource: h
Nmodel_12_model_11_conv2d_transpose_15_conv2d_transpose_readvariableop_resource: S
Emodel_12_model_11_conv2d_transpose_15_biasadd_readvariableop_resource:
identity??2model_12/model_10/conv2d_10/BiasAdd/ReadVariableOp?1model_12/model_10/conv2d_10/Conv2D/ReadVariableOp?2model_12/model_10/conv2d_11/BiasAdd/ReadVariableOp?1model_12/model_10/conv2d_11/Conv2D/ReadVariableOp?1model_12/model_10/conv2d_8/BiasAdd/ReadVariableOp?0model_12/model_10/conv2d_8/Conv2D/ReadVariableOp?1model_12/model_10/conv2d_9/BiasAdd/ReadVariableOp?0model_12/model_10/conv2d_9/Conv2D/ReadVariableOp?<model_12/model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp?Emodel_12/model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?<model_12/model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp?Emodel_12/model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?<model_12/model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp?Emodel_12/model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?<model_12/model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp?Emodel_12/model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
0model_12/model_10/conv2d_8/Conv2D/ReadVariableOpReadVariableOp9model_12_model_10_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype022
0model_12/model_10/conv2d_8/Conv2D/ReadVariableOp?
!model_12/model_10/conv2d_8/Conv2DConv2Dinput_58model_12/model_10/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2#
!model_12/model_10/conv2d_8/Conv2D?
1model_12/model_10/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp:model_12_model_10_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1model_12/model_10/conv2d_8/BiasAdd/ReadVariableOp?
"model_12/model_10/conv2d_8/BiasAddBiasAdd*model_12/model_10/conv2d_8/Conv2D:output:09model_12/model_10/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2$
"model_12/model_10/conv2d_8/BiasAdd?
model_12/model_10/conv2d_8/ReluRelu+model_12/model_10/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2!
model_12/model_10/conv2d_8/Relu?
)model_12/model_10/max_pooling2d_4/MaxPoolMaxPool-model_12/model_10/conv2d_8/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2+
)model_12/model_10/max_pooling2d_4/MaxPool?
0model_12/model_10/conv2d_9/Conv2D/ReadVariableOpReadVariableOp9model_12_model_10_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype022
0model_12/model_10/conv2d_9/Conv2D/ReadVariableOp?
!model_12/model_10/conv2d_9/Conv2DConv2D2model_12/model_10/max_pooling2d_4/MaxPool:output:08model_12/model_10/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2#
!model_12/model_10/conv2d_9/Conv2D?
1model_12/model_10/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp:model_12_model_10_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1model_12/model_10/conv2d_9/BiasAdd/ReadVariableOp?
"model_12/model_10/conv2d_9/BiasAddBiasAdd*model_12/model_10/conv2d_9/Conv2D:output:09model_12/model_10/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"model_12/model_10/conv2d_9/BiasAdd?
model_12/model_10/conv2d_9/ReluRelu+model_12/model_10/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2!
model_12/model_10/conv2d_9/Relu?
)model_12/model_10/max_pooling2d_5/MaxPoolMaxPool-model_12/model_10/conv2d_9/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2+
)model_12/model_10/max_pooling2d_5/MaxPool?
1model_12/model_10/conv2d_10/Conv2D/ReadVariableOpReadVariableOp:model_12_model_10_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1model_12/model_10/conv2d_10/Conv2D/ReadVariableOp?
"model_12/model_10/conv2d_10/Conv2DConv2D2model_12/model_10/max_pooling2d_5/MaxPool:output:09model_12/model_10/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2$
"model_12/model_10/conv2d_10/Conv2D?
2model_12/model_10/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp;model_12_model_10_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2model_12/model_10/conv2d_10/BiasAdd/ReadVariableOp?
#model_12/model_10/conv2d_10/BiasAddBiasAdd+model_12/model_10/conv2d_10/Conv2D:output:0:model_12/model_10/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2%
#model_12/model_10/conv2d_10/BiasAdd?
 model_12/model_10/conv2d_10/ReluRelu,model_12/model_10/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2"
 model_12/model_10/conv2d_10/Relu?
1model_12/model_10/conv2d_11/Conv2D/ReadVariableOpReadVariableOp:model_12_model_10_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype023
1model_12/model_10/conv2d_11/Conv2D/ReadVariableOp?
"model_12/model_10/conv2d_11/Conv2DConv2D.model_12/model_10/conv2d_10/Relu:activations:09model_12/model_10/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2$
"model_12/model_10/conv2d_11/Conv2D?
2model_12/model_10/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp;model_12_model_10_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2model_12/model_10/conv2d_11/BiasAdd/ReadVariableOp?
#model_12/model_10/conv2d_11/BiasAddBiasAdd+model_12/model_10/conv2d_11/Conv2D:output:0:model_12/model_10/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2%
#model_12/model_10/conv2d_11/BiasAdd?
 model_12/model_10/conv2d_11/ReluRelu,model_12/model_10/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2"
 model_12/model_10/conv2d_11/Relu?
!model_12/model_10/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2#
!model_12/model_10/flatten_2/Const?
#model_12/model_10/flatten_2/ReshapeReshape.model_12/model_10/conv2d_11/Relu:activations:0*model_12/model_10/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2%
#model_12/model_10/flatten_2/Reshape?
!model_12/model_11/reshape_3/ShapeShape,model_12/model_10/flatten_2/Reshape:output:0*
T0*
_output_shapes
:2#
!model_12/model_11/reshape_3/Shape?
/model_12/model_11/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/model_12/model_11/reshape_3/strided_slice/stack?
1model_12/model_11/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1model_12/model_11/reshape_3/strided_slice/stack_1?
1model_12/model_11/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1model_12/model_11/reshape_3/strided_slice/stack_2?
)model_12/model_11/reshape_3/strided_sliceStridedSlice*model_12/model_11/reshape_3/Shape:output:08model_12/model_11/reshape_3/strided_slice/stack:output:0:model_12/model_11/reshape_3/strided_slice/stack_1:output:0:model_12/model_11/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)model_12/model_11/reshape_3/strided_slice?
+model_12/model_11/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+model_12/model_11/reshape_3/Reshape/shape/1?
+model_12/model_11/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+model_12/model_11/reshape_3/Reshape/shape/2?
+model_12/model_11/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2-
+model_12/model_11/reshape_3/Reshape/shape/3?
)model_12/model_11/reshape_3/Reshape/shapePack2model_12/model_11/reshape_3/strided_slice:output:04model_12/model_11/reshape_3/Reshape/shape/1:output:04model_12/model_11/reshape_3/Reshape/shape/2:output:04model_12/model_11/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)model_12/model_11/reshape_3/Reshape/shape?
#model_12/model_11/reshape_3/ReshapeReshape,model_12/model_10/flatten_2/Reshape:output:02model_12/model_11/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2%
#model_12/model_11/reshape_3/Reshape?
+model_12/model_11/conv2d_transpose_12/ShapeShape,model_12/model_11/reshape_3/Reshape:output:0*
T0*
_output_shapes
:2-
+model_12/model_11/conv2d_transpose_12/Shape?
9model_12/model_11/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model_12/model_11/conv2d_transpose_12/strided_slice/stack?
;model_12/model_11/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;model_12/model_11/conv2d_transpose_12/strided_slice/stack_1?
;model_12/model_11/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;model_12/model_11/conv2d_transpose_12/strided_slice/stack_2?
3model_12/model_11/conv2d_transpose_12/strided_sliceStridedSlice4model_12/model_11/conv2d_transpose_12/Shape:output:0Bmodel_12/model_11/conv2d_transpose_12/strided_slice/stack:output:0Dmodel_12/model_11/conv2d_transpose_12/strided_slice/stack_1:output:0Dmodel_12/model_11/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3model_12/model_11/conv2d_transpose_12/strided_slice?
-model_12/model_11/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_12/stack/1?
-model_12/model_11/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_12/stack/2?
-model_12/model_11/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_12/stack/3?
+model_12/model_11/conv2d_transpose_12/stackPack<model_12/model_11/conv2d_transpose_12/strided_slice:output:06model_12/model_11/conv2d_transpose_12/stack/1:output:06model_12/model_11/conv2d_transpose_12/stack/2:output:06model_12/model_11/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2-
+model_12/model_11/conv2d_transpose_12/stack?
;model_12/model_11/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;model_12/model_11/conv2d_transpose_12/strided_slice_1/stack?
=model_12/model_11/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=model_12/model_11/conv2d_transpose_12/strided_slice_1/stack_1?
=model_12/model_11/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=model_12/model_11/conv2d_transpose_12/strided_slice_1/stack_2?
5model_12/model_11/conv2d_transpose_12/strided_slice_1StridedSlice4model_12/model_11/conv2d_transpose_12/stack:output:0Dmodel_12/model_11/conv2d_transpose_12/strided_slice_1/stack:output:0Fmodel_12/model_11/conv2d_transpose_12/strided_slice_1/stack_1:output:0Fmodel_12/model_11/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5model_12/model_11/conv2d_transpose_12/strided_slice_1?
Emodel_12/model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpNmodel_12_model_11_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02G
Emodel_12/model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
6model_12/model_11/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput4model_12/model_11/conv2d_transpose_12/stack:output:0Mmodel_12/model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0,model_12/model_11/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
28
6model_12/model_11/conv2d_transpose_12/conv2d_transpose?
<model_12/model_11/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOpEmodel_12_model_11_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<model_12/model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp?
-model_12/model_11/conv2d_transpose_12/BiasAddBiasAdd?model_12/model_11/conv2d_transpose_12/conv2d_transpose:output:0Dmodel_12/model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2/
-model_12/model_11/conv2d_transpose_12/BiasAdd?
*model_12/model_11/conv2d_transpose_12/ReluRelu6model_12/model_11/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2,
*model_12/model_11/conv2d_transpose_12/Relu?
+model_12/model_11/conv2d_transpose_13/ShapeShape8model_12/model_11/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2-
+model_12/model_11/conv2d_transpose_13/Shape?
9model_12/model_11/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model_12/model_11/conv2d_transpose_13/strided_slice/stack?
;model_12/model_11/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;model_12/model_11/conv2d_transpose_13/strided_slice/stack_1?
;model_12/model_11/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;model_12/model_11/conv2d_transpose_13/strided_slice/stack_2?
3model_12/model_11/conv2d_transpose_13/strided_sliceStridedSlice4model_12/model_11/conv2d_transpose_13/Shape:output:0Bmodel_12/model_11/conv2d_transpose_13/strided_slice/stack:output:0Dmodel_12/model_11/conv2d_transpose_13/strided_slice/stack_1:output:0Dmodel_12/model_11/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3model_12/model_11/conv2d_transpose_13/strided_slice?
-model_12/model_11/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_13/stack/1?
-model_12/model_11/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_13/stack/2?
-model_12/model_11/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_13/stack/3?
+model_12/model_11/conv2d_transpose_13/stackPack<model_12/model_11/conv2d_transpose_13/strided_slice:output:06model_12/model_11/conv2d_transpose_13/stack/1:output:06model_12/model_11/conv2d_transpose_13/stack/2:output:06model_12/model_11/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2-
+model_12/model_11/conv2d_transpose_13/stack?
;model_12/model_11/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;model_12/model_11/conv2d_transpose_13/strided_slice_1/stack?
=model_12/model_11/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=model_12/model_11/conv2d_transpose_13/strided_slice_1/stack_1?
=model_12/model_11/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=model_12/model_11/conv2d_transpose_13/strided_slice_1/stack_2?
5model_12/model_11/conv2d_transpose_13/strided_slice_1StridedSlice4model_12/model_11/conv2d_transpose_13/stack:output:0Dmodel_12/model_11/conv2d_transpose_13/strided_slice_1/stack:output:0Fmodel_12/model_11/conv2d_transpose_13/strided_slice_1/stack_1:output:0Fmodel_12/model_11/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5model_12/model_11/conv2d_transpose_13/strided_slice_1?
Emodel_12/model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpNmodel_12_model_11_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02G
Emodel_12/model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
6model_12/model_11/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput4model_12/model_11/conv2d_transpose_13/stack:output:0Mmodel_12/model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:08model_12/model_11/conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
28
6model_12/model_11/conv2d_transpose_13/conv2d_transpose?
<model_12/model_11/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOpEmodel_12_model_11_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<model_12/model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp?
-model_12/model_11/conv2d_transpose_13/BiasAddBiasAdd?model_12/model_11/conv2d_transpose_13/conv2d_transpose:output:0Dmodel_12/model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2/
-model_12/model_11/conv2d_transpose_13/BiasAdd?
*model_12/model_11/conv2d_transpose_13/ReluRelu6model_12/model_11/conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2,
*model_12/model_11/conv2d_transpose_13/Relu?
'model_12/model_11/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2)
'model_12/model_11/up_sampling2d_6/Const?
)model_12/model_11/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)model_12/model_11/up_sampling2d_6/Const_1?
%model_12/model_11/up_sampling2d_6/mulMul0model_12/model_11/up_sampling2d_6/Const:output:02model_12/model_11/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2'
%model_12/model_11/up_sampling2d_6/mul?
>model_12/model_11/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor8model_12/model_11/conv2d_transpose_13/Relu:activations:0)model_12/model_11/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2@
>model_12/model_11/up_sampling2d_6/resize/ResizeNearestNeighbor?
+model_12/model_11/conv2d_transpose_14/ShapeShapeOmodel_12/model_11/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2-
+model_12/model_11/conv2d_transpose_14/Shape?
9model_12/model_11/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model_12/model_11/conv2d_transpose_14/strided_slice/stack?
;model_12/model_11/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;model_12/model_11/conv2d_transpose_14/strided_slice/stack_1?
;model_12/model_11/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;model_12/model_11/conv2d_transpose_14/strided_slice/stack_2?
3model_12/model_11/conv2d_transpose_14/strided_sliceStridedSlice4model_12/model_11/conv2d_transpose_14/Shape:output:0Bmodel_12/model_11/conv2d_transpose_14/strided_slice/stack:output:0Dmodel_12/model_11/conv2d_transpose_14/strided_slice/stack_1:output:0Dmodel_12/model_11/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3model_12/model_11/conv2d_transpose_14/strided_slice?
-model_12/model_11/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_14/stack/1?
-model_12/model_11/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_14/stack/2?
-model_12/model_11/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2/
-model_12/model_11/conv2d_transpose_14/stack/3?
+model_12/model_11/conv2d_transpose_14/stackPack<model_12/model_11/conv2d_transpose_14/strided_slice:output:06model_12/model_11/conv2d_transpose_14/stack/1:output:06model_12/model_11/conv2d_transpose_14/stack/2:output:06model_12/model_11/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2-
+model_12/model_11/conv2d_transpose_14/stack?
;model_12/model_11/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;model_12/model_11/conv2d_transpose_14/strided_slice_1/stack?
=model_12/model_11/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=model_12/model_11/conv2d_transpose_14/strided_slice_1/stack_1?
=model_12/model_11/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=model_12/model_11/conv2d_transpose_14/strided_slice_1/stack_2?
5model_12/model_11/conv2d_transpose_14/strided_slice_1StridedSlice4model_12/model_11/conv2d_transpose_14/stack:output:0Dmodel_12/model_11/conv2d_transpose_14/strided_slice_1/stack:output:0Fmodel_12/model_11/conv2d_transpose_14/strided_slice_1/stack_1:output:0Fmodel_12/model_11/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5model_12/model_11/conv2d_transpose_14/strided_slice_1?
Emodel_12/model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpNmodel_12_model_11_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02G
Emodel_12/model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
6model_12/model_11/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput4model_12/model_11/conv2d_transpose_14/stack:output:0Mmodel_12/model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0Omodel_12/model_11/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
28
6model_12/model_11/conv2d_transpose_14/conv2d_transpose?
<model_12/model_11/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOpEmodel_12_model_11_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02>
<model_12/model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp?
-model_12/model_11/conv2d_transpose_14/BiasAddBiasAdd?model_12/model_11/conv2d_transpose_14/conv2d_transpose:output:0Dmodel_12/model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2/
-model_12/model_11/conv2d_transpose_14/BiasAdd?
*model_12/model_11/conv2d_transpose_14/ReluRelu6model_12/model_11/conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2,
*model_12/model_11/conv2d_transpose_14/Relu?
'model_12/model_11/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2)
'model_12/model_11/up_sampling2d_7/Const?
)model_12/model_11/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2+
)model_12/model_11/up_sampling2d_7/Const_1?
%model_12/model_11/up_sampling2d_7/mulMul0model_12/model_11/up_sampling2d_7/Const:output:02model_12/model_11/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2'
%model_12/model_11/up_sampling2d_7/mul?
>model_12/model_11/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor8model_12/model_11/conv2d_transpose_14/Relu:activations:0)model_12/model_11/up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2@
>model_12/model_11/up_sampling2d_7/resize/ResizeNearestNeighbor?
+model_12/model_11/conv2d_transpose_15/ShapeShapeOmodel_12/model_11/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2-
+model_12/model_11/conv2d_transpose_15/Shape?
9model_12/model_11/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9model_12/model_11/conv2d_transpose_15/strided_slice/stack?
;model_12/model_11/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;model_12/model_11/conv2d_transpose_15/strided_slice/stack_1?
;model_12/model_11/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;model_12/model_11/conv2d_transpose_15/strided_slice/stack_2?
3model_12/model_11/conv2d_transpose_15/strided_sliceStridedSlice4model_12/model_11/conv2d_transpose_15/Shape:output:0Bmodel_12/model_11/conv2d_transpose_15/strided_slice/stack:output:0Dmodel_12/model_11/conv2d_transpose_15/strided_slice/stack_1:output:0Dmodel_12/model_11/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3model_12/model_11/conv2d_transpose_15/strided_slice?
-model_12/model_11/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_15/stack/1?
-model_12/model_11/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_15/stack/2?
-model_12/model_11/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-model_12/model_11/conv2d_transpose_15/stack/3?
+model_12/model_11/conv2d_transpose_15/stackPack<model_12/model_11/conv2d_transpose_15/strided_slice:output:06model_12/model_11/conv2d_transpose_15/stack/1:output:06model_12/model_11/conv2d_transpose_15/stack/2:output:06model_12/model_11/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2-
+model_12/model_11/conv2d_transpose_15/stack?
;model_12/model_11/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2=
;model_12/model_11/conv2d_transpose_15/strided_slice_1/stack?
=model_12/model_11/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2?
=model_12/model_11/conv2d_transpose_15/strided_slice_1/stack_1?
=model_12/model_11/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2?
=model_12/model_11/conv2d_transpose_15/strided_slice_1/stack_2?
5model_12/model_11/conv2d_transpose_15/strided_slice_1StridedSlice4model_12/model_11/conv2d_transpose_15/stack:output:0Dmodel_12/model_11/conv2d_transpose_15/strided_slice_1/stack:output:0Fmodel_12/model_11/conv2d_transpose_15/strided_slice_1/stack_1:output:0Fmodel_12/model_11/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask27
5model_12/model_11/conv2d_transpose_15/strided_slice_1?
Emodel_12/model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpNmodel_12_model_11_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02G
Emodel_12/model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
6model_12/model_11/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput4model_12/model_11/conv2d_transpose_15/stack:output:0Mmodel_12/model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0Omodel_12/model_11/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
28
6model_12/model_11/conv2d_transpose_15/conv2d_transpose?
<model_12/model_11/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOpEmodel_12_model_11_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<model_12/model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp?
-model_12/model_11/conv2d_transpose_15/BiasAddBiasAdd?model_12/model_11/conv2d_transpose_15/conv2d_transpose:output:0Dmodel_12/model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2/
-model_12/model_11/conv2d_transpose_15/BiasAdd?
-model_12/model_11/conv2d_transpose_15/SigmoidSigmoid6model_12/model_11/conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2/
-model_12/model_11/conv2d_transpose_15/Sigmoid?
IdentityIdentity1model_12/model_11/conv2d_transpose_15/Sigmoid:y:03^model_12/model_10/conv2d_10/BiasAdd/ReadVariableOp2^model_12/model_10/conv2d_10/Conv2D/ReadVariableOp3^model_12/model_10/conv2d_11/BiasAdd/ReadVariableOp2^model_12/model_10/conv2d_11/Conv2D/ReadVariableOp2^model_12/model_10/conv2d_8/BiasAdd/ReadVariableOp1^model_12/model_10/conv2d_8/Conv2D/ReadVariableOp2^model_12/model_10/conv2d_9/BiasAdd/ReadVariableOp1^model_12/model_10/conv2d_9/Conv2D/ReadVariableOp=^model_12/model_11/conv2d_transpose_12/BiasAdd/ReadVariableOpF^model_12/model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp=^model_12/model_11/conv2d_transpose_13/BiasAdd/ReadVariableOpF^model_12/model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp=^model_12/model_11/conv2d_transpose_14/BiasAdd/ReadVariableOpF^model_12/model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp=^model_12/model_11/conv2d_transpose_15/BiasAdd/ReadVariableOpF^model_12/model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2h
2model_12/model_10/conv2d_10/BiasAdd/ReadVariableOp2model_12/model_10/conv2d_10/BiasAdd/ReadVariableOp2f
1model_12/model_10/conv2d_10/Conv2D/ReadVariableOp1model_12/model_10/conv2d_10/Conv2D/ReadVariableOp2h
2model_12/model_10/conv2d_11/BiasAdd/ReadVariableOp2model_12/model_10/conv2d_11/BiasAdd/ReadVariableOp2f
1model_12/model_10/conv2d_11/Conv2D/ReadVariableOp1model_12/model_10/conv2d_11/Conv2D/ReadVariableOp2f
1model_12/model_10/conv2d_8/BiasAdd/ReadVariableOp1model_12/model_10/conv2d_8/BiasAdd/ReadVariableOp2d
0model_12/model_10/conv2d_8/Conv2D/ReadVariableOp0model_12/model_10/conv2d_8/Conv2D/ReadVariableOp2f
1model_12/model_10/conv2d_9/BiasAdd/ReadVariableOp1model_12/model_10/conv2d_9/BiasAdd/ReadVariableOp2d
0model_12/model_10/conv2d_9/Conv2D/ReadVariableOp0model_12/model_10/conv2d_9/Conv2D/ReadVariableOp2|
<model_12/model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp<model_12/model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp2?
Emodel_12/model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOpEmodel_12/model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2|
<model_12/model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp<model_12/model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp2?
Emodel_12/model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOpEmodel_12/model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2|
<model_12/model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp<model_12/model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp2?
Emodel_12/model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOpEmodel_12/model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2|
<model_12/model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp<model_12/model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp2?
Emodel_12/model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOpEmodel_12/model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?0
?
C__inference_model_10_layer_call_and_return_conditional_losses_21460

inputsA
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource: A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource:
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_8/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_8/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling2d_4/MaxPool?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_9/Relu?
max_pooling2d_5/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_5/MaxPool?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_10/Relu?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_11/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_2/Const?
flatten_2/ReshapeReshapeconv2d_11/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
IdentityIdentityflatten_2/Reshape:output:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_5_layer_call_fn_19965

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_199592
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_9_layer_call_fn_21771

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_200012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
(__inference_model_11_layer_call_fn_21538

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_11_layer_call_and_return_conditional_losses_205992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_model_12_layer_call_and_return_conditional_losses_20735

inputs(
model_10_20700: 
model_10_20702: (
model_10_20704: 
model_10_20706:(
model_10_20708:
model_10_20710:(
model_10_20712:
model_10_20714:(
model_11_20717:
model_11_20719:(
model_11_20721:
model_11_20723:(
model_11_20725: 
model_11_20727: (
model_11_20729: 
model_11_20731:
identity?? model_10/StatefulPartitionedCall? model_11/StatefulPartitionedCall?
 model_10/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_10_20700model_10_20702model_10_20704model_10_20706model_10_20708model_10_20710model_10_20712model_10_20714*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_200512"
 model_10/StatefulPartitionedCall?
 model_11/StatefulPartitionedCallStatefulPartitionedCall)model_10/StatefulPartitionedCall:output:0model_11_20717model_11_20719model_11_20721model_11_20723model_11_20725model_11_20727model_11_20729model_11_20731*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_11_layer_call_and_return_conditional_losses_205242"
 model_11/StatefulPartitionedCall?
IdentityIdentity)model_11/StatefulPartitionedCall:output:0!^model_10/StatefulPartitionedCall!^model_11/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2D
 model_10/StatefulPartitionedCall model_10/StatefulPartitionedCall2D
 model_11/StatefulPartitionedCall model_11/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_20363

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_10_layer_call_fn_21791

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_200192
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_21762

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
(__inference_model_11_layer_call_fn_21517

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_11_layer_call_and_return_conditional_losses_205242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_21782

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_21040
input_5!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_199412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?&
?
C__inference_model_11_layer_call_and_return_conditional_losses_20666
input_63
conv2d_transpose_12_20643:'
conv2d_transpose_12_20645:3
conv2d_transpose_13_20648:'
conv2d_transpose_13_20650:3
conv2d_transpose_14_20654: '
conv2d_transpose_14_20656: 3
conv2d_transpose_15_20660: '
conv2d_transpose_15_20662:
identity??+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?
reshape_3/PartitionedCallPartitionedCallinput_6*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204992
reshape_3/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_12_20643conv2d_transpose_12_20645*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_202952-
+conv2d_transpose_12/StatefulPartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_20648conv2d_transpose_13_20650*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_203402-
+conv2d_transpose_13/StatefulPartitionedCall?
up_sampling2d_6/PartitionedCallPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_203632!
up_sampling2d_6/PartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_14_20654conv2d_transpose_14_20656*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_204042-
+conv2d_transpose_14/StatefulPartitionedCall?
up_sampling2d_7/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_204272!
up_sampling2d_7/PartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0conv2d_transpose_15_20660conv2d_transpose_15_20662*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_204682-
+conv2d_transpose_15/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_15/StatefulPartitionedCall:output:0,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_6
??
?%
!__inference__traced_restore_22215
file_prefix$
assignvariableop_adam_iter:	 (
assignvariableop_1_adam_beta_1: (
assignvariableop_2_adam_beta_2: '
assignvariableop_3_adam_decay: /
%assignvariableop_4_adam_learning_rate: <
"assignvariableop_5_conv2d_8_kernel: .
 assignvariableop_6_conv2d_8_bias: <
"assignvariableop_7_conv2d_9_kernel: .
 assignvariableop_8_conv2d_9_bias:=
#assignvariableop_9_conv2d_10_kernel:0
"assignvariableop_10_conv2d_10_bias:>
$assignvariableop_11_conv2d_11_kernel:0
"assignvariableop_12_conv2d_11_bias:H
.assignvariableop_13_conv2d_transpose_12_kernel::
,assignvariableop_14_conv2d_transpose_12_bias:H
.assignvariableop_15_conv2d_transpose_13_kernel::
,assignvariableop_16_conv2d_transpose_13_bias:H
.assignvariableop_17_conv2d_transpose_14_kernel: :
,assignvariableop_18_conv2d_transpose_14_bias: H
.assignvariableop_19_conv2d_transpose_15_kernel: :
,assignvariableop_20_conv2d_transpose_15_bias:#
assignvariableop_21_total: #
assignvariableop_22_count: D
*assignvariableop_23_adam_conv2d_8_kernel_m: 6
(assignvariableop_24_adam_conv2d_8_bias_m: D
*assignvariableop_25_adam_conv2d_9_kernel_m: 6
(assignvariableop_26_adam_conv2d_9_bias_m:E
+assignvariableop_27_adam_conv2d_10_kernel_m:7
)assignvariableop_28_adam_conv2d_10_bias_m:E
+assignvariableop_29_adam_conv2d_11_kernel_m:7
)assignvariableop_30_adam_conv2d_11_bias_m:O
5assignvariableop_31_adam_conv2d_transpose_12_kernel_m:A
3assignvariableop_32_adam_conv2d_transpose_12_bias_m:O
5assignvariableop_33_adam_conv2d_transpose_13_kernel_m:A
3assignvariableop_34_adam_conv2d_transpose_13_bias_m:O
5assignvariableop_35_adam_conv2d_transpose_14_kernel_m: A
3assignvariableop_36_adam_conv2d_transpose_14_bias_m: O
5assignvariableop_37_adam_conv2d_transpose_15_kernel_m: A
3assignvariableop_38_adam_conv2d_transpose_15_bias_m:D
*assignvariableop_39_adam_conv2d_8_kernel_v: 6
(assignvariableop_40_adam_conv2d_8_bias_v: D
*assignvariableop_41_adam_conv2d_9_kernel_v: 6
(assignvariableop_42_adam_conv2d_9_bias_v:E
+assignvariableop_43_adam_conv2d_10_kernel_v:7
)assignvariableop_44_adam_conv2d_10_bias_v:E
+assignvariableop_45_adam_conv2d_11_kernel_v:7
)assignvariableop_46_adam_conv2d_11_bias_v:O
5assignvariableop_47_adam_conv2d_transpose_12_kernel_v:A
3assignvariableop_48_adam_conv2d_transpose_12_bias_v:O
5assignvariableop_49_adam_conv2d_transpose_13_kernel_v:A
3assignvariableop_50_adam_conv2d_transpose_13_bias_v:O
5assignvariableop_51_adam_conv2d_transpose_14_kernel_v: A
3assignvariableop_52_adam_conv2d_transpose_14_bias_v: O
5assignvariableop_53_adam_conv2d_transpose_15_kernel_v: A
3assignvariableop_54_adam_conv2d_transpose_15_bias_v:
identity_56??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_conv2d_8_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp assignvariableop_6_conv2d_8_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp"assignvariableop_7_conv2d_9_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_conv2d_9_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp#assignvariableop_9_conv2d_10_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp"assignvariableop_10_conv2d_10_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp$assignvariableop_11_conv2d_11_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp"assignvariableop_12_conv2d_11_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp.assignvariableop_13_conv2d_transpose_12_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp,assignvariableop_14_conv2d_transpose_12_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp.assignvariableop_15_conv2d_transpose_13_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp,assignvariableop_16_conv2d_transpose_13_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp.assignvariableop_17_conv2d_transpose_14_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp,assignvariableop_18_conv2d_transpose_14_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp.assignvariableop_19_conv2d_transpose_15_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp,assignvariableop_20_conv2d_transpose_15_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_8_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_8_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_adam_conv2d_9_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp(assignvariableop_26_adam_conv2d_9_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_10_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_10_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_11_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_11_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp5assignvariableop_31_adam_conv2d_transpose_12_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp3assignvariableop_32_adam_conv2d_transpose_12_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp5assignvariableop_33_adam_conv2d_transpose_13_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_conv2d_transpose_13_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_adam_conv2d_transpose_14_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp3assignvariableop_36_adam_conv2d_transpose_14_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp5assignvariableop_37_adam_conv2d_transpose_15_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp3assignvariableop_38_adam_conv2d_transpose_15_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_8_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_8_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_9_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_9_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_10_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_10_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_11_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_11_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp5assignvariableop_47_adam_conv2d_transpose_12_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp3assignvariableop_48_adam_conv2d_transpose_12_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp5assignvariableop_49_adam_conv2d_transpose_13_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp3assignvariableop_50_adam_conv2d_transpose_13_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_conv2d_transpose_14_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_conv2d_transpose_14_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_conv2d_transpose_15_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp3assignvariableop_54_adam_conv2d_transpose_15_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55?

Identity_56IdentityIdentity_55:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_56"#
identity_56Identity_56:output:0*?
_input_shapesr
p: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_20019

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_reshape_3_layer_call_fn_21838

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_4_layer_call_fn_19953

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_199472
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?&
?
C__inference_model_11_layer_call_and_return_conditional_losses_20599

inputs3
conv2d_transpose_12_20576:'
conv2d_transpose_12_20578:3
conv2d_transpose_13_20581:'
conv2d_transpose_13_20583:3
conv2d_transpose_14_20587: '
conv2d_transpose_14_20589: 3
conv2d_transpose_15_20593: '
conv2d_transpose_15_20595:
identity??+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?
reshape_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204992
reshape_3/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_12_20576conv2d_transpose_12_20578*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_202952-
+conv2d_transpose_12/StatefulPartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_20581conv2d_transpose_13_20583*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_203402-
+conv2d_transpose_13/StatefulPartitionedCall?
up_sampling2d_6/PartitionedCallPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_203632!
up_sampling2d_6/PartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_14_20587conv2d_transpose_14_20589*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_204042-
+conv2d_transpose_14/StatefulPartitionedCall?
up_sampling2d_7/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_204272!
up_sampling2d_7/PartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0conv2d_transpose_15_20593conv2d_transpose_15_20595*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_204682-
+conv2d_transpose_15/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_15/StatefulPartitionedCall:output:0,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_19983

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
(__inference_model_10_layer_call_fn_21424

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_201662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_3_layer_call_and_return_conditional_losses_21852

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_7_layer_call_fn_20433

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_204272
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_model_12_layer_call_fn_20770
input_5!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_207352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
?
3__inference_conv2d_transpose_15_layer_call_fn_20478

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_204682
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_20001

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
(__inference_model_11_layer_call_fn_20543
input_6!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_11_layer_call_and_return_conditional_losses_205242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_6
?
?
)__inference_conv2d_11_layer_call_fn_21811

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_200362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
C__inference_model_10_layer_call_and_return_conditional_losses_20051

inputs(
conv2d_8_19984: 
conv2d_8_19986: (
conv2d_9_20002: 
conv2d_9_20004:)
conv2d_10_20020:
conv2d_10_20022:)
conv2d_11_20037:
conv2d_11_20039:
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_19984conv2d_8_19986*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_199832"
 conv2d_8/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_199472!
max_pooling2d_4/PartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_9_20002conv2d_9_20004*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_200012"
 conv2d_9/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_199592!
max_pooling2d_5/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_10_20020conv2d_10_20022*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_200192#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_20037conv2d_11_20039*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_200362#
!conv2d_11/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_200482
flatten_2/PartitionedCall?
IdentityIdentity"flatten_2/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
C__inference_model_11_layer_call_and_return_conditional_losses_20524

inputs3
conv2d_transpose_12_20501:'
conv2d_transpose_12_20503:3
conv2d_transpose_13_20506:'
conv2d_transpose_13_20508:3
conv2d_transpose_14_20512: '
conv2d_transpose_14_20514: 3
conv2d_transpose_15_20518: '
conv2d_transpose_15_20520:
identity??+conv2d_transpose_12/StatefulPartitionedCall?+conv2d_transpose_13/StatefulPartitionedCall?+conv2d_transpose_14/StatefulPartitionedCall?+conv2d_transpose_15/StatefulPartitionedCall?
reshape_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_reshape_3_layer_call_and_return_conditional_losses_204992
reshape_3/PartitionedCall?
+conv2d_transpose_12/StatefulPartitionedCallStatefulPartitionedCall"reshape_3/PartitionedCall:output:0conv2d_transpose_12_20501conv2d_transpose_12_20503*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_202952-
+conv2d_transpose_12/StatefulPartitionedCall?
+conv2d_transpose_13/StatefulPartitionedCallStatefulPartitionedCall4conv2d_transpose_12/StatefulPartitionedCall:output:0conv2d_transpose_13_20506conv2d_transpose_13_20508*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_203402-
+conv2d_transpose_13/StatefulPartitionedCall?
up_sampling2d_6/PartitionedCallPartitionedCall4conv2d_transpose_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_203632!
up_sampling2d_6/PartitionedCall?
+conv2d_transpose_14/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_6/PartitionedCall:output:0conv2d_transpose_14_20512conv2d_transpose_14_20514*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_204042-
+conv2d_transpose_14/StatefulPartitionedCall?
up_sampling2d_7/PartitionedCallPartitionedCall4conv2d_transpose_14/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_204272!
up_sampling2d_7/PartitionedCall?
+conv2d_transpose_15/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_7/PartitionedCall:output:0conv2d_transpose_15_20518conv2d_transpose_15_20520*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_204682-
+conv2d_transpose_15/StatefulPartitionedCall?
IdentityIdentity4conv2d_transpose_15/StatefulPartitionedCall:output:0,^conv2d_transpose_12/StatefulPartitionedCall,^conv2d_transpose_13/StatefulPartitionedCall,^conv2d_transpose_14/StatefulPartitionedCall,^conv2d_transpose_15/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2Z
+conv2d_transpose_12/StatefulPartitionedCall+conv2d_transpose_12/StatefulPartitionedCall2Z
+conv2d_transpose_13/StatefulPartitionedCall+conv2d_transpose_13/StatefulPartitionedCall2Z
+conv2d_transpose_14/StatefulPartitionedCall+conv2d_transpose_14/StatefulPartitionedCall2Z
+conv2d_transpose_15/StatefulPartitionedCall+conv2d_transpose_15/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_12_layer_call_fn_20919
input_5!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_208472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
?
3__inference_conv2d_transpose_12_layer_call_fn_20305

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_202952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?%
?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_20404

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?p
?
__inference__traced_save_22040
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop.
*savev2_conv2d_9_kernel_read_readvariableop,
(savev2_conv2d_9_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop/
+savev2_conv2d_11_kernel_read_readvariableop-
)savev2_conv2d_11_bias_read_readvariableop9
5savev2_conv2d_transpose_12_kernel_read_readvariableop7
3savev2_conv2d_transpose_12_bias_read_readvariableop9
5savev2_conv2d_transpose_13_kernel_read_readvariableop7
3savev2_conv2d_transpose_13_bias_read_readvariableop9
5savev2_conv2d_transpose_14_kernel_read_readvariableop7
3savev2_conv2d_transpose_14_bias_read_readvariableop9
5savev2_conv2d_transpose_15_kernel_read_readvariableop7
3savev2_conv2d_transpose_15_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableop5
1savev2_adam_conv2d_9_kernel_m_read_readvariableop3
/savev2_adam_conv2d_9_bias_m_read_readvariableop6
2savev2_adam_conv2d_10_kernel_m_read_readvariableop4
0savev2_adam_conv2d_10_bias_m_read_readvariableop6
2savev2_adam_conv2d_11_kernel_m_read_readvariableop4
0savev2_adam_conv2d_11_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_12_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_12_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_13_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_13_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_14_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_14_bias_m_read_readvariableop@
<savev2_adam_conv2d_transpose_15_kernel_m_read_readvariableop>
:savev2_adam_conv2d_transpose_15_bias_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableop5
1savev2_adam_conv2d_9_kernel_v_read_readvariableop3
/savev2_adam_conv2d_9_bias_v_read_readvariableop6
2savev2_adam_conv2d_10_kernel_v_read_readvariableop4
0savev2_adam_conv2d_10_bias_v_read_readvariableop6
2savev2_adam_conv2d_11_kernel_v_read_readvariableop4
0savev2_adam_conv2d_11_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_12_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_12_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_13_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_13_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_14_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_14_bias_v_read_readvariableop@
<savev2_adam_conv2d_transpose_15_kernel_v_read_readvariableop>
:savev2_adam_conv2d_transpose_15_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop*savev2_conv2d_9_kernel_read_readvariableop(savev2_conv2d_9_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop+savev2_conv2d_11_kernel_read_readvariableop)savev2_conv2d_11_bias_read_readvariableop5savev2_conv2d_transpose_12_kernel_read_readvariableop3savev2_conv2d_transpose_12_bias_read_readvariableop5savev2_conv2d_transpose_13_kernel_read_readvariableop3savev2_conv2d_transpose_13_bias_read_readvariableop5savev2_conv2d_transpose_14_kernel_read_readvariableop3savev2_conv2d_transpose_14_bias_read_readvariableop5savev2_conv2d_transpose_15_kernel_read_readvariableop3savev2_conv2d_transpose_15_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop1savev2_adam_conv2d_9_kernel_m_read_readvariableop/savev2_adam_conv2d_9_bias_m_read_readvariableop2savev2_adam_conv2d_10_kernel_m_read_readvariableop0savev2_adam_conv2d_10_bias_m_read_readvariableop2savev2_adam_conv2d_11_kernel_m_read_readvariableop0savev2_adam_conv2d_11_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_12_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_12_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_13_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_13_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_14_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_14_bias_m_read_readvariableop<savev2_adam_conv2d_transpose_15_kernel_m_read_readvariableop:savev2_adam_conv2d_transpose_15_bias_m_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableop1savev2_adam_conv2d_9_kernel_v_read_readvariableop/savev2_adam_conv2d_9_bias_v_read_readvariableop2savev2_adam_conv2d_10_kernel_v_read_readvariableop0savev2_adam_conv2d_10_bias_v_read_readvariableop2savev2_adam_conv2d_11_kernel_v_read_readvariableop0savev2_adam_conv2d_11_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_12_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_12_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_13_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_13_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_14_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_14_bias_v_read_readvariableop<savev2_adam_conv2d_transpose_15_kernel_v_read_readvariableop:savev2_adam_conv2d_transpose_15_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : : :::::::::: : : :: : : : : :::::::::: : : :: : : :::::::::: : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 	

_output_shapes
::,
(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:: #

_output_shapes
::,$(
&
_output_shapes
: : %

_output_shapes
: :,&(
&
_output_shapes
: : '

_output_shapes
::,((
&
_output_shapes
: : )

_output_shapes
: :,*(
&
_output_shapes
: : +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
: : 5

_output_shapes
: :,6(
&
_output_shapes
: : 7

_output_shapes
::8

_output_shapes
: 
?
?
3__inference_conv2d_transpose_13_layer_call_fn_20350

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_203402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_20048

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
C__inference_model_10_layer_call_and_return_conditional_losses_20233
input_5(
conv2d_8_20209: 
conv2d_8_20211: (
conv2d_9_20215: 
conv2d_9_20217:)
conv2d_10_20221:
conv2d_10_20223:)
conv2d_11_20226:
conv2d_11_20228:
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_8_20209conv2d_8_20211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_199832"
 conv2d_8/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_199472!
max_pooling2d_4/PartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_9_20215conv2d_9_20217*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_200012"
 conv2d_9/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_199592!
max_pooling2d_5/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_10_20221conv2d_10_20223*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_200192#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_20226conv2d_11_20228*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_200362#
!conv2d_11/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_200482
flatten_2/PartitionedCall?
IdentityIdentity"flatten_2/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?0
?
C__inference_model_10_layer_call_and_return_conditional_losses_21496

inputsA
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource: A
'conv2d_9_conv2d_readvariableop_resource: 6
(conv2d_9_biasadd_readvariableop_resource:B
(conv2d_10_conv2d_readvariableop_resource:7
)conv2d_10_biasadd_readvariableop_resource:B
(conv2d_11_conv2d_readvariableop_resource:7
)conv2d_11_biasadd_readvariableop_resource:
identity?? conv2d_10/BiasAdd/ReadVariableOp?conv2d_10/Conv2D/ReadVariableOp? conv2d_11/BiasAdd/ReadVariableOp?conv2d_11/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?conv2d_9/BiasAdd/ReadVariableOp?conv2d_9/Conv2D/ReadVariableOp?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dinputs&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_8/Relu?
max_pooling2d_4/MaxPoolMaxPoolconv2d_8/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling2d_4/MaxPool?
conv2d_9/Conv2D/ReadVariableOpReadVariableOp'conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_9/Conv2D/ReadVariableOp?
conv2d_9/Conv2DConv2D max_pooling2d_4/MaxPool:output:0&conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_9/Conv2D?
conv2d_9/BiasAdd/ReadVariableOpReadVariableOp(conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_9/BiasAdd/ReadVariableOp?
conv2d_9/BiasAddBiasAddconv2d_9/Conv2D:output:0'conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_9/BiasAdd{
conv2d_9/ReluReluconv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_9/Relu?
max_pooling2d_5/MaxPoolMaxPoolconv2d_9/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_5/MaxPool?
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp?
conv2d_10/Conv2DConv2D max_pooling2d_5/MaxPool:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_10/Conv2D?
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp?
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_10/Relu?
conv2d_11/Conv2D/ReadVariableOpReadVariableOp(conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_11/Conv2D/ReadVariableOp?
conv2d_11/Conv2DConv2Dconv2d_10/Relu:activations:0'conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_11/Conv2D?
 conv2d_11/BiasAdd/ReadVariableOpReadVariableOp)conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_11/BiasAdd/ReadVariableOp?
conv2d_11/BiasAddBiasAddconv2d_11/Conv2D:output:0(conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_11/BiasAdd~
conv2d_11/ReluReluconv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_11/Relus
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
flatten_2/Const?
flatten_2/ReshapeReshapeconv2d_11/Relu:activations:0flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten_2/Reshape?
IdentityIdentityflatten_2/Reshape:output:0!^conv2d_10/BiasAdd/ReadVariableOp ^conv2d_10/Conv2D/ReadVariableOp!^conv2d_11/BiasAdd/ReadVariableOp ^conv2d_11/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp ^conv2d_9/BiasAdd/ReadVariableOp^conv2d_9/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2D
 conv2d_10/BiasAdd/ReadVariableOp conv2d_10/BiasAdd/ReadVariableOp2B
conv2d_10/Conv2D/ReadVariableOpconv2d_10/Conv2D/ReadVariableOp2D
 conv2d_11/BiasAdd/ReadVariableOp conv2d_11/BiasAdd/ReadVariableOp2B
conv2d_11/Conv2D/ReadVariableOpconv2d_11/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2B
conv2d_9/BiasAdd/ReadVariableOpconv2d_9/BiasAdd/ReadVariableOp2@
conv2d_9/Conv2D/ReadVariableOpconv2d_9/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
C__inference_model_11_layer_call_and_return_conditional_losses_21640

inputsV
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_12_biasadd_readvariableop_resource:V
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_13_biasadd_readvariableop_resource:V
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_14_biasadd_readvariableop_resource: V
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_15_biasadd_readvariableop_resource:
identity??*conv2d_transpose_12/BiasAdd/ReadVariableOp?3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?*conv2d_transpose_13/BiasAdd/ReadVariableOp?3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?*conv2d_transpose_14/BiasAdd/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?*conv2d_transpose_15/BiasAdd/ReadVariableOp?3conv2d_transpose_15/conv2d_transpose/ReadVariableOpX
reshape_3/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_3/Shape?
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stack?
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1?
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2x
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/3?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape?
reshape_3/ReshapeReshapeinputs reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_3/Reshape?
conv2d_transpose_12/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_12/Shape?
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_12/strided_slice/stack?
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_1?
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_2?
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_12/strided_slice|
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/1|
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/2|
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/3?
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_12/stack?
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_12/strided_slice_1/stack?
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_1?
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_2?
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_12/strided_slice_1?
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_12/conv2d_transpose?
*conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_12/BiasAdd/ReadVariableOp?
conv2d_transpose_12/BiasAddBiasAdd-conv2d_transpose_12/conv2d_transpose:output:02conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_12/BiasAdd?
conv2d_transpose_12/ReluRelu$conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_12/Relu?
conv2d_transpose_13/ShapeShape&conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_13/Shape?
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_13/strided_slice/stack?
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_1?
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_2?
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_13/strided_slice|
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/1|
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/2|
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/3?
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_13/stack?
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_13/strided_slice_1/stack?
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_1?
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_2?
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_13/strided_slice_1?
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_13/conv2d_transpose?
*conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_13/BiasAdd/ReadVariableOp?
conv2d_transpose_13/BiasAddBiasAdd-conv2d_transpose_13/conv2d_transpose:output:02conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_13/BiasAdd?
conv2d_transpose_13/ReluRelu$conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_13/Relu
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const?
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const_1?
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul?
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor&conv2d_transpose_13/Relu:activations:0up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor?
conv2d_transpose_14/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_14/Shape?
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_14/strided_slice/stack?
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_1?
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_2?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_14/strided_slice|
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/1|
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/2|
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_14/stack/3?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_14/stack?
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_14/strided_slice_1/stack?
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_1?
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_2?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_14/strided_slice_1?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2&
$conv2d_transpose_14/conv2d_transpose?
*conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_14/BiasAdd/ReadVariableOp?
conv2d_transpose_14/BiasAddBiasAdd-conv2d_transpose_14/conv2d_transpose:output:02conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_14/BiasAdd?
conv2d_transpose_14/ReluRelu$conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_14/Relu
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const?
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const_1?
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul?
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor&conv2d_transpose_14/Relu:activations:0up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor?
conv2d_transpose_15/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_15/Shape?
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_15/strided_slice/stack?
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_1?
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_2?
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_15/strided_slice|
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/1|
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/2|
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/3?
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_15/stack?
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_15/strided_slice_1/stack?
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_1?
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_2?
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_15/strided_slice_1?
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_15/conv2d_transpose?
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_15/BiasAdd/ReadVariableOp?
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_15/BiasAdd?
conv2d_transpose_15/SigmoidSigmoid$conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_15/Sigmoid?
IdentityIdentityconv2d_transpose_15/Sigmoid:y:0+^conv2d_transpose_12/BiasAdd/ReadVariableOp4^conv2d_transpose_12/conv2d_transpose/ReadVariableOp+^conv2d_transpose_13/BiasAdd/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp+^conv2d_transpose_14/BiasAdd/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2X
*conv2d_transpose_12/BiasAdd/ReadVariableOp*conv2d_transpose_12/BiasAdd/ReadVariableOp2j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_13/BiasAdd/ReadVariableOp*conv2d_transpose_13/BiasAdd/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_14/BiasAdd/ReadVariableOp*conv2d_transpose_14/BiasAdd/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
C__inference_model_12_layer_call_and_return_conditional_losses_20957
input_5(
model_10_20922: 
model_10_20924: (
model_10_20926: 
model_10_20928:(
model_10_20930:
model_10_20932:(
model_10_20934:
model_10_20936:(
model_11_20939:
model_11_20941:(
model_11_20943:
model_11_20945:(
model_11_20947: 
model_11_20949: (
model_11_20951: 
model_11_20953:
identity?? model_10/StatefulPartitionedCall? model_11/StatefulPartitionedCall?
 model_10/StatefulPartitionedCallStatefulPartitionedCallinput_5model_10_20922model_10_20924model_10_20926model_10_20928model_10_20930model_10_20932model_10_20934model_10_20936*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_200512"
 model_10/StatefulPartitionedCall?
 model_11/StatefulPartitionedCallStatefulPartitionedCall)model_10/StatefulPartitionedCall:output:0model_11_20939model_11_20941model_11_20943model_11_20945model_11_20947model_11_20949model_11_20951model_11_20953*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_11_layer_call_and_return_conditional_losses_205242"
 model_11/StatefulPartitionedCall?
IdentityIdentity)model_11/StatefulPartitionedCall:output:0!^model_10/StatefulPartitionedCall!^model_11/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2D
 model_10/StatefulPartitionedCall model_10/StatefulPartitionedCall2D
 model_11/StatefulPartitionedCall model_11/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
??
?
C__inference_model_11_layer_call_and_return_conditional_losses_21742

inputsV
<conv2d_transpose_12_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_12_biasadd_readvariableop_resource:V
<conv2d_transpose_13_conv2d_transpose_readvariableop_resource:A
3conv2d_transpose_13_biasadd_readvariableop_resource:V
<conv2d_transpose_14_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_14_biasadd_readvariableop_resource: V
<conv2d_transpose_15_conv2d_transpose_readvariableop_resource: A
3conv2d_transpose_15_biasadd_readvariableop_resource:
identity??*conv2d_transpose_12/BiasAdd/ReadVariableOp?3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?*conv2d_transpose_13/BiasAdd/ReadVariableOp?3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?*conv2d_transpose_14/BiasAdd/ReadVariableOp?3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?*conv2d_transpose_15/BiasAdd/ReadVariableOp?3conv2d_transpose_15/conv2d_transpose/ReadVariableOpX
reshape_3/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape_3/Shape?
reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_3/strided_slice/stack?
reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_1?
reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_3/strided_slice/stack_2?
reshape_3/strided_sliceStridedSlicereshape_3/Shape:output:0&reshape_3/strided_slice/stack:output:0(reshape_3/strided_slice/stack_1:output:0(reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_3/strided_slicex
reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/1x
reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/2x
reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_3/Reshape/shape/3?
reshape_3/Reshape/shapePack reshape_3/strided_slice:output:0"reshape_3/Reshape/shape/1:output:0"reshape_3/Reshape/shape/2:output:0"reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_3/Reshape/shape?
reshape_3/ReshapeReshapeinputs reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_3/Reshape?
conv2d_transpose_12/ShapeShapereshape_3/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose_12/Shape?
'conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_12/strided_slice/stack?
)conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_1?
)conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_12/strided_slice/stack_2?
!conv2d_transpose_12/strided_sliceStridedSlice"conv2d_transpose_12/Shape:output:00conv2d_transpose_12/strided_slice/stack:output:02conv2d_transpose_12/strided_slice/stack_1:output:02conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_12/strided_slice|
conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/1|
conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/2|
conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_12/stack/3?
conv2d_transpose_12/stackPack*conv2d_transpose_12/strided_slice:output:0$conv2d_transpose_12/stack/1:output:0$conv2d_transpose_12/stack/2:output:0$conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_12/stack?
)conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_12/strided_slice_1/stack?
+conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_1?
+conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_12/strided_slice_1/stack_2?
#conv2d_transpose_12/strided_slice_1StridedSlice"conv2d_transpose_12/stack:output:02conv2d_transpose_12/strided_slice_1/stack:output:04conv2d_transpose_12/strided_slice_1/stack_1:output:04conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_12/strided_slice_1?
3conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_12/conv2d_transposeConv2DBackpropInput"conv2d_transpose_12/stack:output:0;conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_12/conv2d_transpose?
*conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_12/BiasAdd/ReadVariableOp?
conv2d_transpose_12/BiasAddBiasAdd-conv2d_transpose_12/conv2d_transpose:output:02conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_12/BiasAdd?
conv2d_transpose_12/ReluRelu$conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_12/Relu?
conv2d_transpose_13/ShapeShape&conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_13/Shape?
'conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_13/strided_slice/stack?
)conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_1?
)conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_13/strided_slice/stack_2?
!conv2d_transpose_13/strided_sliceStridedSlice"conv2d_transpose_13/Shape:output:00conv2d_transpose_13/strided_slice/stack:output:02conv2d_transpose_13/strided_slice/stack_1:output:02conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_13/strided_slice|
conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/1|
conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/2|
conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_13/stack/3?
conv2d_transpose_13/stackPack*conv2d_transpose_13/strided_slice:output:0$conv2d_transpose_13/stack/1:output:0$conv2d_transpose_13/stack/2:output:0$conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_13/stack?
)conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_13/strided_slice_1/stack?
+conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_1?
+conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_13/strided_slice_1/stack_2?
#conv2d_transpose_13/strided_slice_1StridedSlice"conv2d_transpose_13/stack:output:02conv2d_transpose_13/strided_slice_1/stack:output:04conv2d_transpose_13/strided_slice_1/stack_1:output:04conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_13/strided_slice_1?
3conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype025
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_13/conv2d_transposeConv2DBackpropInput"conv2d_transpose_13/stack:output:0;conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0&conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_13/conv2d_transpose?
*conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_13/BiasAdd/ReadVariableOp?
conv2d_transpose_13/BiasAddBiasAdd-conv2d_transpose_13/conv2d_transpose:output:02conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_13/BiasAdd?
conv2d_transpose_13/ReluRelu$conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_13/Relu
up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const?
up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_6/Const_1?
up_sampling2d_6/mulMulup_sampling2d_6/Const:output:0 up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_6/mul?
,up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor&conv2d_transpose_13/Relu:activations:0up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2.
,up_sampling2d_6/resize/ResizeNearestNeighbor?
conv2d_transpose_14/ShapeShape=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_14/Shape?
'conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_14/strided_slice/stack?
)conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_1?
)conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_14/strided_slice/stack_2?
!conv2d_transpose_14/strided_sliceStridedSlice"conv2d_transpose_14/Shape:output:00conv2d_transpose_14/strided_slice/stack:output:02conv2d_transpose_14/strided_slice/stack_1:output:02conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_14/strided_slice|
conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/1|
conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_14/stack/2|
conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_14/stack/3?
conv2d_transpose_14/stackPack*conv2d_transpose_14/strided_slice:output:0$conv2d_transpose_14/stack/1:output:0$conv2d_transpose_14/stack/2:output:0$conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_14/stack?
)conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_14/strided_slice_1/stack?
+conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_1?
+conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_14/strided_slice_1/stack_2?
#conv2d_transpose_14/strided_slice_1StridedSlice"conv2d_transpose_14/stack:output:02conv2d_transpose_14/strided_slice_1/stack:output:04conv2d_transpose_14/strided_slice_1/stack_1:output:04conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_14/strided_slice_1?
3conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_14/conv2d_transposeConv2DBackpropInput"conv2d_transpose_14/stack:output:0;conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2&
$conv2d_transpose_14/conv2d_transpose?
*conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*conv2d_transpose_14/BiasAdd/ReadVariableOp?
conv2d_transpose_14/BiasAddBiasAdd-conv2d_transpose_14/conv2d_transpose:output:02conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_14/BiasAdd?
conv2d_transpose_14/ReluRelu$conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_14/Relu
up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const?
up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_7/Const_1?
up_sampling2d_7/mulMulup_sampling2d_7/Const:output:0 up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_7/mul?
,up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor&conv2d_transpose_14/Relu:activations:0up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2.
,up_sampling2d_7/resize/ResizeNearestNeighbor?
conv2d_transpose_15/ShapeShape=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2
conv2d_transpose_15/Shape?
'conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2)
'conv2d_transpose_15/strided_slice/stack?
)conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_1?
)conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)conv2d_transpose_15/strided_slice/stack_2?
!conv2d_transpose_15/strided_sliceStridedSlice"conv2d_transpose_15/Shape:output:00conv2d_transpose_15/strided_slice/stack:output:02conv2d_transpose_15/strided_slice/stack_1:output:02conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!conv2d_transpose_15/strided_slice|
conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/1|
conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/2|
conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_15/stack/3?
conv2d_transpose_15/stackPack*conv2d_transpose_15/strided_slice:output:0$conv2d_transpose_15/stack/1:output:0$conv2d_transpose_15/stack/2:output:0$conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_15/stack?
)conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)conv2d_transpose_15/strided_slice_1/stack?
+conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_1?
+conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+conv2d_transpose_15/strided_slice_1/stack_2?
#conv2d_transpose_15/strided_slice_1StridedSlice"conv2d_transpose_15/stack:output:02conv2d_transpose_15/strided_slice_1/stack:output:04conv2d_transpose_15/strided_slice_1/stack_1:output:04conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#conv2d_transpose_15/strided_slice_1?
3conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOp<conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype025
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
$conv2d_transpose_15/conv2d_transposeConv2DBackpropInput"conv2d_transpose_15/stack:output:0;conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0=up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2&
$conv2d_transpose_15/conv2d_transpose?
*conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp3conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*conv2d_transpose_15/BiasAdd/ReadVariableOp?
conv2d_transpose_15/BiasAddBiasAdd-conv2d_transpose_15/conv2d_transpose:output:02conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_15/BiasAdd?
conv2d_transpose_15/SigmoidSigmoid$conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_15/Sigmoid?
IdentityIdentityconv2d_transpose_15/Sigmoid:y:0+^conv2d_transpose_12/BiasAdd/ReadVariableOp4^conv2d_transpose_12/conv2d_transpose/ReadVariableOp+^conv2d_transpose_13/BiasAdd/ReadVariableOp4^conv2d_transpose_13/conv2d_transpose/ReadVariableOp+^conv2d_transpose_14/BiasAdd/ReadVariableOp4^conv2d_transpose_14/conv2d_transpose/ReadVariableOp+^conv2d_transpose_15/BiasAdd/ReadVariableOp4^conv2d_transpose_15/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2X
*conv2d_transpose_12/BiasAdd/ReadVariableOp*conv2d_transpose_12/BiasAdd/ReadVariableOp2j
3conv2d_transpose_12/conv2d_transpose/ReadVariableOp3conv2d_transpose_12/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_13/BiasAdd/ReadVariableOp*conv2d_transpose_13/BiasAdd/ReadVariableOp2j
3conv2d_transpose_13/conv2d_transpose/ReadVariableOp3conv2d_transpose_13/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_14/BiasAdd/ReadVariableOp*conv2d_transpose_14/BiasAdd/ReadVariableOp2j
3conv2d_transpose_14/conv2d_transpose/ReadVariableOp3conv2d_transpose_14/conv2d_transpose/ReadVariableOp2X
*conv2d_transpose_15/BiasAdd/ReadVariableOp*conv2d_transpose_15/BiasAdd/ReadVariableOp2j
3conv2d_transpose_15/conv2d_transpose/ReadVariableOp3conv2d_transpose_15/conv2d_transpose/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
C__inference_model_10_layer_call_and_return_conditional_losses_20166

inputs(
conv2d_8_20142: 
conv2d_8_20144: (
conv2d_9_20148: 
conv2d_9_20150:)
conv2d_10_20154:
conv2d_10_20156:)
conv2d_11_20159:
conv2d_11_20161:
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_8_20142conv2d_8_20144*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_199832"
 conv2d_8/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_199472!
max_pooling2d_4/PartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_9_20148conv2d_9_20150*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_200012"
 conv2d_9/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_199592!
max_pooling2d_5/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_10_20154conv2d_10_20156*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_200192#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_20159conv2d_11_20161*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_200362#
!conv2d_11/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_200482
flatten_2/PartitionedCall?
IdentityIdentity"flatten_2/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
(__inference_model_11_layer_call_fn_20639
input_6!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3: 
	unknown_4: #
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_11_layer_call_and_return_conditional_losses_205992
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_6
?	
?
(__inference_model_10_layer_call_fn_20206
input_5!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_201662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_20036

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?!
?
C__inference_model_10_layer_call_and_return_conditional_losses_20260
input_5(
conv2d_8_20236: 
conv2d_8_20238: (
conv2d_9_20242: 
conv2d_9_20244:)
conv2d_10_20248:
conv2d_10_20250:)
conv2d_11_20253:
conv2d_11_20255:
identity??!conv2d_10/StatefulPartitionedCall?!conv2d_11/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall? conv2d_9/StatefulPartitionedCall?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_8_20236conv2d_8_20238*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_199832"
 conv2d_8/StatefulPartitionedCall?
max_pooling2d_4/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_199472!
max_pooling2d_4/PartitionedCall?
 conv2d_9/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_4/PartitionedCall:output:0conv2d_9_20242conv2d_9_20244*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_9_layer_call_and_return_conditional_losses_200012"
 conv2d_9/StatefulPartitionedCall?
max_pooling2d_5/PartitionedCallPartitionedCall)conv2d_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_199592!
max_pooling2d_5/PartitionedCall?
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_5/PartitionedCall:output:0conv2d_10_20248conv2d_10_20250*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_200192#
!conv2d_10/StatefulPartitionedCall?
!conv2d_11/StatefulPartitionedCallStatefulPartitionedCall*conv2d_10/StatefulPartitionedCall:output:0conv2d_11_20253conv2d_11_20255*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_11_layer_call_and_return_conditional_losses_200362#
!conv2d_11/StatefulPartitionedCall?
flatten_2/PartitionedCallPartitionedCall*conv2d_11/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_200482
flatten_2/PartitionedCall?
IdentityIdentity"flatten_2/PartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall"^conv2d_11/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall!^conv2d_9/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2F
!conv2d_11/StatefulPartitionedCall!conv2d_11/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2D
 conv2d_9/StatefulPartitionedCall conv2d_9/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
?
?
C__inference_model_12_layer_call_and_return_conditional_losses_20847

inputs(
model_10_20812: 
model_10_20814: (
model_10_20816: 
model_10_20818:(
model_10_20820:
model_10_20822:(
model_10_20824:
model_10_20826:(
model_11_20829:
model_11_20831:(
model_11_20833:
model_11_20835:(
model_11_20837: 
model_11_20839: (
model_11_20841: 
model_11_20843:
identity?? model_10/StatefulPartitionedCall? model_11/StatefulPartitionedCall?
 model_10/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_10_20812model_10_20814model_10_20816model_10_20818model_10_20820model_10_20822model_10_20824model_10_20826*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_201662"
 model_10/StatefulPartitionedCall?
 model_11/StatefulPartitionedCallStatefulPartitionedCall)model_10/StatefulPartitionedCall:output:0model_11_20829model_11_20831model_11_20833model_11_20835model_11_20837model_11_20839model_11_20841model_11_20843*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_11_layer_call_and_return_conditional_losses_205992"
 model_11/StatefulPartitionedCall?
IdentityIdentity)model_11/StatefulPartitionedCall:output:0!^model_10/StatefulPartitionedCall!^model_11/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2D
 model_10/StatefulPartitionedCall model_10/StatefulPartitionedCall2D
 model_11/StatefulPartitionedCall model_11/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_20427

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_2_layer_call_fn_21827

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_flatten_2_layer_call_and_return_conditional_losses_200482
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
C__inference_model_12_layer_call_and_return_conditional_losses_21248

inputsJ
0model_10_conv2d_8_conv2d_readvariableop_resource: ?
1model_10_conv2d_8_biasadd_readvariableop_resource: J
0model_10_conv2d_9_conv2d_readvariableop_resource: ?
1model_10_conv2d_9_biasadd_readvariableop_resource:K
1model_10_conv2d_10_conv2d_readvariableop_resource:@
2model_10_conv2d_10_biasadd_readvariableop_resource:K
1model_10_conv2d_11_conv2d_readvariableop_resource:@
2model_10_conv2d_11_biasadd_readvariableop_resource:_
Emodel_11_conv2d_transpose_12_conv2d_transpose_readvariableop_resource:J
<model_11_conv2d_transpose_12_biasadd_readvariableop_resource:_
Emodel_11_conv2d_transpose_13_conv2d_transpose_readvariableop_resource:J
<model_11_conv2d_transpose_13_biasadd_readvariableop_resource:_
Emodel_11_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: J
<model_11_conv2d_transpose_14_biasadd_readvariableop_resource: _
Emodel_11_conv2d_transpose_15_conv2d_transpose_readvariableop_resource: J
<model_11_conv2d_transpose_15_biasadd_readvariableop_resource:
identity??)model_10/conv2d_10/BiasAdd/ReadVariableOp?(model_10/conv2d_10/Conv2D/ReadVariableOp?)model_10/conv2d_11/BiasAdd/ReadVariableOp?(model_10/conv2d_11/Conv2D/ReadVariableOp?(model_10/conv2d_8/BiasAdd/ReadVariableOp?'model_10/conv2d_8/Conv2D/ReadVariableOp?(model_10/conv2d_9/BiasAdd/ReadVariableOp?'model_10/conv2d_9/Conv2D/ReadVariableOp?3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp?<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp?<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp?<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp?<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
'model_10/conv2d_8/Conv2D/ReadVariableOpReadVariableOp0model_10_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_10/conv2d_8/Conv2D/ReadVariableOp?
model_10/conv2d_8/Conv2DConv2Dinputs/model_10/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
model_10/conv2d_8/Conv2D?
(model_10/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp1model_10_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_10/conv2d_8/BiasAdd/ReadVariableOp?
model_10/conv2d_8/BiasAddBiasAdd!model_10/conv2d_8/Conv2D:output:00model_10/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
model_10/conv2d_8/BiasAdd?
model_10/conv2d_8/ReluRelu"model_10/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
model_10/conv2d_8/Relu?
 model_10/max_pooling2d_4/MaxPoolMaxPool$model_10/conv2d_8/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2"
 model_10/max_pooling2d_4/MaxPool?
'model_10/conv2d_9/Conv2D/ReadVariableOpReadVariableOp0model_10_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_10/conv2d_9/Conv2D/ReadVariableOp?
model_10/conv2d_9/Conv2DConv2D)model_10/max_pooling2d_4/MaxPool:output:0/model_10/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_10/conv2d_9/Conv2D?
(model_10/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp1model_10_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_10/conv2d_9/BiasAdd/ReadVariableOp?
model_10/conv2d_9/BiasAddBiasAdd!model_10/conv2d_9/Conv2D:output:00model_10/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_9/BiasAdd?
model_10/conv2d_9/ReluRelu"model_10/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_9/Relu?
 model_10/max_pooling2d_5/MaxPoolMaxPool$model_10/conv2d_9/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2"
 model_10/max_pooling2d_5/MaxPool?
(model_10/conv2d_10/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(model_10/conv2d_10/Conv2D/ReadVariableOp?
model_10/conv2d_10/Conv2DConv2D)model_10/max_pooling2d_5/MaxPool:output:00model_10/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_10/conv2d_10/Conv2D?
)model_10/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_10/conv2d_10/BiasAdd/ReadVariableOp?
model_10/conv2d_10/BiasAddBiasAdd"model_10/conv2d_10/Conv2D:output:01model_10/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_10/BiasAdd?
model_10/conv2d_10/ReluRelu#model_10/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_10/Relu?
(model_10/conv2d_11/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(model_10/conv2d_11/Conv2D/ReadVariableOp?
model_10/conv2d_11/Conv2DConv2D%model_10/conv2d_10/Relu:activations:00model_10/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_10/conv2d_11/Conv2D?
)model_10/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_10/conv2d_11/BiasAdd/ReadVariableOp?
model_10/conv2d_11/BiasAddBiasAdd"model_10/conv2d_11/Conv2D:output:01model_10/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_11/BiasAdd?
model_10/conv2d_11/ReluRelu#model_10/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_11/Relu?
model_10/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
model_10/flatten_2/Const?
model_10/flatten_2/ReshapeReshape%model_10/conv2d_11/Relu:activations:0!model_10/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
model_10/flatten_2/Reshape?
model_11/reshape_3/ShapeShape#model_10/flatten_2/Reshape:output:0*
T0*
_output_shapes
:2
model_11/reshape_3/Shape?
&model_11/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_11/reshape_3/strided_slice/stack?
(model_11/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_11/reshape_3/strided_slice/stack_1?
(model_11/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_11/reshape_3/strided_slice/stack_2?
 model_11/reshape_3/strided_sliceStridedSlice!model_11/reshape_3/Shape:output:0/model_11/reshape_3/strided_slice/stack:output:01model_11/reshape_3/strided_slice/stack_1:output:01model_11/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_11/reshape_3/strided_slice?
"model_11/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_11/reshape_3/Reshape/shape/1?
"model_11/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_11/reshape_3/Reshape/shape/2?
"model_11/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_11/reshape_3/Reshape/shape/3?
 model_11/reshape_3/Reshape/shapePack)model_11/reshape_3/strided_slice:output:0+model_11/reshape_3/Reshape/shape/1:output:0+model_11/reshape_3/Reshape/shape/2:output:0+model_11/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 model_11/reshape_3/Reshape/shape?
model_11/reshape_3/ReshapeReshape#model_10/flatten_2/Reshape:output:0)model_11/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
model_11/reshape_3/Reshape?
"model_11/conv2d_transpose_12/ShapeShape#model_11/reshape_3/Reshape:output:0*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_12/Shape?
0model_11/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_11/conv2d_transpose_12/strided_slice/stack?
2model_11/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_12/strided_slice/stack_1?
2model_11/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_12/strided_slice/stack_2?
*model_11/conv2d_transpose_12/strided_sliceStridedSlice+model_11/conv2d_transpose_12/Shape:output:09model_11/conv2d_transpose_12/strided_slice/stack:output:0;model_11/conv2d_transpose_12/strided_slice/stack_1:output:0;model_11/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_11/conv2d_transpose_12/strided_slice?
$model_11/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_12/stack/1?
$model_11/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_12/stack/2?
$model_11/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_12/stack/3?
"model_11/conv2d_transpose_12/stackPack3model_11/conv2d_transpose_12/strided_slice:output:0-model_11/conv2d_transpose_12/stack/1:output:0-model_11/conv2d_transpose_12/stack/2:output:0-model_11/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_12/stack?
2model_11/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_11/conv2d_transpose_12/strided_slice_1/stack?
4model_11/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_12/strided_slice_1/stack_1?
4model_11/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_12/strided_slice_1/stack_2?
,model_11/conv2d_transpose_12/strided_slice_1StridedSlice+model_11/conv2d_transpose_12/stack:output:0;model_11/conv2d_transpose_12/strided_slice_1/stack:output:0=model_11/conv2d_transpose_12/strided_slice_1/stack_1:output:0=model_11/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_11/conv2d_transpose_12/strided_slice_1?
<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_11_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02>
<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
-model_11/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput+model_11/conv2d_transpose_12/stack:output:0Dmodel_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0#model_11/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-model_11/conv2d_transpose_12/conv2d_transpose?
3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp<model_11_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp?
$model_11/conv2d_transpose_12/BiasAddBiasAdd6model_11/conv2d_transpose_12/conv2d_transpose:output:0;model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2&
$model_11/conv2d_transpose_12/BiasAdd?
!model_11/conv2d_transpose_12/ReluRelu-model_11/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2#
!model_11/conv2d_transpose_12/Relu?
"model_11/conv2d_transpose_13/ShapeShape/model_11/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_13/Shape?
0model_11/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_11/conv2d_transpose_13/strided_slice/stack?
2model_11/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_13/strided_slice/stack_1?
2model_11/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_13/strided_slice/stack_2?
*model_11/conv2d_transpose_13/strided_sliceStridedSlice+model_11/conv2d_transpose_13/Shape:output:09model_11/conv2d_transpose_13/strided_slice/stack:output:0;model_11/conv2d_transpose_13/strided_slice/stack_1:output:0;model_11/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_11/conv2d_transpose_13/strided_slice?
$model_11/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_13/stack/1?
$model_11/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_13/stack/2?
$model_11/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_13/stack/3?
"model_11/conv2d_transpose_13/stackPack3model_11/conv2d_transpose_13/strided_slice:output:0-model_11/conv2d_transpose_13/stack/1:output:0-model_11/conv2d_transpose_13/stack/2:output:0-model_11/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_13/stack?
2model_11/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_11/conv2d_transpose_13/strided_slice_1/stack?
4model_11/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_13/strided_slice_1/stack_1?
4model_11/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_13/strided_slice_1/stack_2?
,model_11/conv2d_transpose_13/strided_slice_1StridedSlice+model_11/conv2d_transpose_13/stack:output:0;model_11/conv2d_transpose_13/strided_slice_1/stack:output:0=model_11/conv2d_transpose_13/strided_slice_1/stack_1:output:0=model_11/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_11/conv2d_transpose_13/strided_slice_1?
<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_11_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02>
<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
-model_11/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput+model_11/conv2d_transpose_13/stack:output:0Dmodel_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0/model_11/conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-model_11/conv2d_transpose_13/conv2d_transpose?
3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp<model_11_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp?
$model_11/conv2d_transpose_13/BiasAddBiasAdd6model_11/conv2d_transpose_13/conv2d_transpose:output:0;model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2&
$model_11/conv2d_transpose_13/BiasAdd?
!model_11/conv2d_transpose_13/ReluRelu-model_11/conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2#
!model_11/conv2d_transpose_13/Relu?
model_11/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2 
model_11/up_sampling2d_6/Const?
 model_11/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model_11/up_sampling2d_6/Const_1?
model_11/up_sampling2d_6/mulMul'model_11/up_sampling2d_6/Const:output:0)model_11/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2
model_11/up_sampling2d_6/mul?
5model_11/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor/model_11/conv2d_transpose_13/Relu:activations:0 model_11/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(27
5model_11/up_sampling2d_6/resize/ResizeNearestNeighbor?
"model_11/conv2d_transpose_14/ShapeShapeFmodel_11/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_14/Shape?
0model_11/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_11/conv2d_transpose_14/strided_slice/stack?
2model_11/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_14/strided_slice/stack_1?
2model_11/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_14/strided_slice/stack_2?
*model_11/conv2d_transpose_14/strided_sliceStridedSlice+model_11/conv2d_transpose_14/Shape:output:09model_11/conv2d_transpose_14/strided_slice/stack:output:0;model_11/conv2d_transpose_14/strided_slice/stack_1:output:0;model_11/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_11/conv2d_transpose_14/strided_slice?
$model_11/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_14/stack/1?
$model_11/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_14/stack/2?
$model_11/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2&
$model_11/conv2d_transpose_14/stack/3?
"model_11/conv2d_transpose_14/stackPack3model_11/conv2d_transpose_14/strided_slice:output:0-model_11/conv2d_transpose_14/stack/1:output:0-model_11/conv2d_transpose_14/stack/2:output:0-model_11/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_14/stack?
2model_11/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_11/conv2d_transpose_14/strided_slice_1/stack?
4model_11/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_14/strided_slice_1/stack_1?
4model_11/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_14/strided_slice_1/stack_2?
,model_11/conv2d_transpose_14/strided_slice_1StridedSlice+model_11/conv2d_transpose_14/stack:output:0;model_11/conv2d_transpose_14/strided_slice_1/stack:output:0=model_11/conv2d_transpose_14/strided_slice_1/stack_1:output:0=model_11/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_11/conv2d_transpose_14/strided_slice_1?
<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_11_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02>
<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
-model_11/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput+model_11/conv2d_transpose_14/stack:output:0Dmodel_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0Fmodel_11/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2/
-model_11/conv2d_transpose_14/conv2d_transpose?
3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp<model_11_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp?
$model_11/conv2d_transpose_14/BiasAddBiasAdd6model_11/conv2d_transpose_14/conv2d_transpose:output:0;model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2&
$model_11/conv2d_transpose_14/BiasAdd?
!model_11/conv2d_transpose_14/ReluRelu-model_11/conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2#
!model_11/conv2d_transpose_14/Relu?
model_11/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2 
model_11/up_sampling2d_7/Const?
 model_11/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model_11/up_sampling2d_7/Const_1?
model_11/up_sampling2d_7/mulMul'model_11/up_sampling2d_7/Const:output:0)model_11/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2
model_11/up_sampling2d_7/mul?
5model_11/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor/model_11/conv2d_transpose_14/Relu:activations:0 model_11/up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(27
5model_11/up_sampling2d_7/resize/ResizeNearestNeighbor?
"model_11/conv2d_transpose_15/ShapeShapeFmodel_11/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_15/Shape?
0model_11/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_11/conv2d_transpose_15/strided_slice/stack?
2model_11/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_15/strided_slice/stack_1?
2model_11/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_15/strided_slice/stack_2?
*model_11/conv2d_transpose_15/strided_sliceStridedSlice+model_11/conv2d_transpose_15/Shape:output:09model_11/conv2d_transpose_15/strided_slice/stack:output:0;model_11/conv2d_transpose_15/strided_slice/stack_1:output:0;model_11/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_11/conv2d_transpose_15/strided_slice?
$model_11/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_15/stack/1?
$model_11/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_15/stack/2?
$model_11/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_15/stack/3?
"model_11/conv2d_transpose_15/stackPack3model_11/conv2d_transpose_15/strided_slice:output:0-model_11/conv2d_transpose_15/stack/1:output:0-model_11/conv2d_transpose_15/stack/2:output:0-model_11/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_15/stack?
2model_11/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_11/conv2d_transpose_15/strided_slice_1/stack?
4model_11/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_15/strided_slice_1/stack_1?
4model_11/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_15/strided_slice_1/stack_2?
,model_11/conv2d_transpose_15/strided_slice_1StridedSlice+model_11/conv2d_transpose_15/stack:output:0;model_11/conv2d_transpose_15/strided_slice_1/stack:output:0=model_11/conv2d_transpose_15/strided_slice_1/stack_1:output:0=model_11/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_11/conv2d_transpose_15/strided_slice_1?
<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_11_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02>
<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
-model_11/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput+model_11/conv2d_transpose_15/stack:output:0Dmodel_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0Fmodel_11/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-model_11/conv2d_transpose_15/conv2d_transpose?
3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp<model_11_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp?
$model_11/conv2d_transpose_15/BiasAddBiasAdd6model_11/conv2d_transpose_15/conv2d_transpose:output:0;model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2&
$model_11/conv2d_transpose_15/BiasAdd?
$model_11/conv2d_transpose_15/SigmoidSigmoid-model_11/conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2&
$model_11/conv2d_transpose_15/Sigmoid?
IdentityIdentity(model_11/conv2d_transpose_15/Sigmoid:y:0*^model_10/conv2d_10/BiasAdd/ReadVariableOp)^model_10/conv2d_10/Conv2D/ReadVariableOp*^model_10/conv2d_11/BiasAdd/ReadVariableOp)^model_10/conv2d_11/Conv2D/ReadVariableOp)^model_10/conv2d_8/BiasAdd/ReadVariableOp(^model_10/conv2d_8/Conv2D/ReadVariableOp)^model_10/conv2d_9/BiasAdd/ReadVariableOp(^model_10/conv2d_9/Conv2D/ReadVariableOp4^model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp=^model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp4^model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp=^model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp4^model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp=^model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp4^model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp=^model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2V
)model_10/conv2d_10/BiasAdd/ReadVariableOp)model_10/conv2d_10/BiasAdd/ReadVariableOp2T
(model_10/conv2d_10/Conv2D/ReadVariableOp(model_10/conv2d_10/Conv2D/ReadVariableOp2V
)model_10/conv2d_11/BiasAdd/ReadVariableOp)model_10/conv2d_11/BiasAdd/ReadVariableOp2T
(model_10/conv2d_11/Conv2D/ReadVariableOp(model_10/conv2d_11/Conv2D/ReadVariableOp2T
(model_10/conv2d_8/BiasAdd/ReadVariableOp(model_10/conv2d_8/BiasAdd/ReadVariableOp2R
'model_10/conv2d_8/Conv2D/ReadVariableOp'model_10/conv2d_8/Conv2D/ReadVariableOp2T
(model_10/conv2d_9/BiasAdd/ReadVariableOp(model_10/conv2d_9/BiasAdd/ReadVariableOp2R
'model_10/conv2d_9/Conv2D/ReadVariableOp'model_10/conv2d_9/Conv2D/ReadVariableOp2j
3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp2|
<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2j
3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp2|
<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2j
3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp2|
<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2j
3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp2|
<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_conv2d_8_layer_call_fn_21751

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_199832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_20340

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
C__inference_model_12_layer_call_and_return_conditional_losses_20995
input_5(
model_10_20960: 
model_10_20962: (
model_10_20964: 
model_10_20966:(
model_10_20968:
model_10_20970:(
model_10_20972:
model_10_20974:(
model_11_20977:
model_11_20979:(
model_11_20981:
model_11_20983:(
model_11_20985: 
model_11_20987: (
model_11_20989: 
model_11_20991:
identity?? model_10/StatefulPartitionedCall? model_11/StatefulPartitionedCall?
 model_10/StatefulPartitionedCallStatefulPartitionedCallinput_5model_10_20960model_10_20962model_10_20964model_10_20966model_10_20968model_10_20970model_10_20972model_10_20974*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_201662"
 model_10/StatefulPartitionedCall?
 model_11/StatefulPartitionedCallStatefulPartitionedCall)model_10/StatefulPartitionedCall:output:0model_11_20977model_11_20979model_11_20981model_11_20983model_11_20985model_11_20987model_11_20989model_11_20991*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_11_layer_call_and_return_conditional_losses_205992"
 model_11/StatefulPartitionedCall?
IdentityIdentity)model_11/StatefulPartitionedCall:output:0!^model_10/StatefulPartitionedCall!^model_11/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2D
 model_10/StatefulPartitionedCall model_10/StatefulPartitionedCall2D
 model_11/StatefulPartitionedCall model_11/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_5
??
?
C__inference_model_12_layer_call_and_return_conditional_losses_21382

inputsJ
0model_10_conv2d_8_conv2d_readvariableop_resource: ?
1model_10_conv2d_8_biasadd_readvariableop_resource: J
0model_10_conv2d_9_conv2d_readvariableop_resource: ?
1model_10_conv2d_9_biasadd_readvariableop_resource:K
1model_10_conv2d_10_conv2d_readvariableop_resource:@
2model_10_conv2d_10_biasadd_readvariableop_resource:K
1model_10_conv2d_11_conv2d_readvariableop_resource:@
2model_10_conv2d_11_biasadd_readvariableop_resource:_
Emodel_11_conv2d_transpose_12_conv2d_transpose_readvariableop_resource:J
<model_11_conv2d_transpose_12_biasadd_readvariableop_resource:_
Emodel_11_conv2d_transpose_13_conv2d_transpose_readvariableop_resource:J
<model_11_conv2d_transpose_13_biasadd_readvariableop_resource:_
Emodel_11_conv2d_transpose_14_conv2d_transpose_readvariableop_resource: J
<model_11_conv2d_transpose_14_biasadd_readvariableop_resource: _
Emodel_11_conv2d_transpose_15_conv2d_transpose_readvariableop_resource: J
<model_11_conv2d_transpose_15_biasadd_readvariableop_resource:
identity??)model_10/conv2d_10/BiasAdd/ReadVariableOp?(model_10/conv2d_10/Conv2D/ReadVariableOp?)model_10/conv2d_11/BiasAdd/ReadVariableOp?(model_10/conv2d_11/Conv2D/ReadVariableOp?(model_10/conv2d_8/BiasAdd/ReadVariableOp?'model_10/conv2d_8/Conv2D/ReadVariableOp?(model_10/conv2d_9/BiasAdd/ReadVariableOp?'model_10/conv2d_9/Conv2D/ReadVariableOp?3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp?<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp?<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp?<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp?<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
'model_10/conv2d_8/Conv2D/ReadVariableOpReadVariableOp0model_10_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_10/conv2d_8/Conv2D/ReadVariableOp?
model_10/conv2d_8/Conv2DConv2Dinputs/model_10/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
model_10/conv2d_8/Conv2D?
(model_10/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp1model_10_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model_10/conv2d_8/BiasAdd/ReadVariableOp?
model_10/conv2d_8/BiasAddBiasAdd!model_10/conv2d_8/Conv2D:output:00model_10/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
model_10/conv2d_8/BiasAdd?
model_10/conv2d_8/ReluRelu"model_10/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
model_10/conv2d_8/Relu?
 model_10/max_pooling2d_4/MaxPoolMaxPool$model_10/conv2d_8/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2"
 model_10/max_pooling2d_4/MaxPool?
'model_10/conv2d_9/Conv2D/ReadVariableOpReadVariableOp0model_10_conv2d_9_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model_10/conv2d_9/Conv2D/ReadVariableOp?
model_10/conv2d_9/Conv2DConv2D)model_10/max_pooling2d_4/MaxPool:output:0/model_10/conv2d_9/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_10/conv2d_9/Conv2D?
(model_10/conv2d_9/BiasAdd/ReadVariableOpReadVariableOp1model_10_conv2d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_10/conv2d_9/BiasAdd/ReadVariableOp?
model_10/conv2d_9/BiasAddBiasAdd!model_10/conv2d_9/Conv2D:output:00model_10/conv2d_9/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_9/BiasAdd?
model_10/conv2d_9/ReluRelu"model_10/conv2d_9/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_9/Relu?
 model_10/max_pooling2d_5/MaxPoolMaxPool$model_10/conv2d_9/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2"
 model_10/max_pooling2d_5/MaxPool?
(model_10/conv2d_10/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(model_10/conv2d_10/Conv2D/ReadVariableOp?
model_10/conv2d_10/Conv2DConv2D)model_10/max_pooling2d_5/MaxPool:output:00model_10/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_10/conv2d_10/Conv2D?
)model_10/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_10/conv2d_10/BiasAdd/ReadVariableOp?
model_10/conv2d_10/BiasAddBiasAdd"model_10/conv2d_10/Conv2D:output:01model_10/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_10/BiasAdd?
model_10/conv2d_10/ReluRelu#model_10/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_10/Relu?
(model_10/conv2d_11/Conv2D/ReadVariableOpReadVariableOp1model_10_conv2d_11_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02*
(model_10/conv2d_11/Conv2D/ReadVariableOp?
model_10/conv2d_11/Conv2DConv2D%model_10/conv2d_10/Relu:activations:00model_10/conv2d_11/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_10/conv2d_11/Conv2D?
)model_10/conv2d_11/BiasAdd/ReadVariableOpReadVariableOp2model_10_conv2d_11_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)model_10/conv2d_11/BiasAdd/ReadVariableOp?
model_10/conv2d_11/BiasAddBiasAdd"model_10/conv2d_11/Conv2D:output:01model_10/conv2d_11/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_11/BiasAdd?
model_10/conv2d_11/ReluRelu#model_10/conv2d_11/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_10/conv2d_11/Relu?
model_10/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
model_10/flatten_2/Const?
model_10/flatten_2/ReshapeReshape%model_10/conv2d_11/Relu:activations:0!model_10/flatten_2/Const:output:0*
T0*(
_output_shapes
:??????????2
model_10/flatten_2/Reshape?
model_11/reshape_3/ShapeShape#model_10/flatten_2/Reshape:output:0*
T0*
_output_shapes
:2
model_11/reshape_3/Shape?
&model_11/reshape_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&model_11/reshape_3/strided_slice/stack?
(model_11/reshape_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_11/reshape_3/strided_slice/stack_1?
(model_11/reshape_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(model_11/reshape_3/strided_slice/stack_2?
 model_11/reshape_3/strided_sliceStridedSlice!model_11/reshape_3/Shape:output:0/model_11/reshape_3/strided_slice/stack:output:01model_11/reshape_3/strided_slice/stack_1:output:01model_11/reshape_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 model_11/reshape_3/strided_slice?
"model_11/reshape_3/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_11/reshape_3/Reshape/shape/1?
"model_11/reshape_3/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_11/reshape_3/Reshape/shape/2?
"model_11/reshape_3/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"model_11/reshape_3/Reshape/shape/3?
 model_11/reshape_3/Reshape/shapePack)model_11/reshape_3/strided_slice:output:0+model_11/reshape_3/Reshape/shape/1:output:0+model_11/reshape_3/Reshape/shape/2:output:0+model_11/reshape_3/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2"
 model_11/reshape_3/Reshape/shape?
model_11/reshape_3/ReshapeReshape#model_10/flatten_2/Reshape:output:0)model_11/reshape_3/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
model_11/reshape_3/Reshape?
"model_11/conv2d_transpose_12/ShapeShape#model_11/reshape_3/Reshape:output:0*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_12/Shape?
0model_11/conv2d_transpose_12/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_11/conv2d_transpose_12/strided_slice/stack?
2model_11/conv2d_transpose_12/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_12/strided_slice/stack_1?
2model_11/conv2d_transpose_12/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_12/strided_slice/stack_2?
*model_11/conv2d_transpose_12/strided_sliceStridedSlice+model_11/conv2d_transpose_12/Shape:output:09model_11/conv2d_transpose_12/strided_slice/stack:output:0;model_11/conv2d_transpose_12/strided_slice/stack_1:output:0;model_11/conv2d_transpose_12/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_11/conv2d_transpose_12/strided_slice?
$model_11/conv2d_transpose_12/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_12/stack/1?
$model_11/conv2d_transpose_12/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_12/stack/2?
$model_11/conv2d_transpose_12/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_12/stack/3?
"model_11/conv2d_transpose_12/stackPack3model_11/conv2d_transpose_12/strided_slice:output:0-model_11/conv2d_transpose_12/stack/1:output:0-model_11/conv2d_transpose_12/stack/2:output:0-model_11/conv2d_transpose_12/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_12/stack?
2model_11/conv2d_transpose_12/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_11/conv2d_transpose_12/strided_slice_1/stack?
4model_11/conv2d_transpose_12/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_12/strided_slice_1/stack_1?
4model_11/conv2d_transpose_12/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_12/strided_slice_1/stack_2?
,model_11/conv2d_transpose_12/strided_slice_1StridedSlice+model_11/conv2d_transpose_12/stack:output:0;model_11/conv2d_transpose_12/strided_slice_1/stack:output:0=model_11/conv2d_transpose_12/strided_slice_1/stack_1:output:0=model_11/conv2d_transpose_12/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_11/conv2d_transpose_12/strided_slice_1?
<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_11_conv2d_transpose_12_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02>
<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp?
-model_11/conv2d_transpose_12/conv2d_transposeConv2DBackpropInput+model_11/conv2d_transpose_12/stack:output:0Dmodel_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp:value:0#model_11/reshape_3/Reshape:output:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-model_11/conv2d_transpose_12/conv2d_transpose?
3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOpReadVariableOp<model_11_conv2d_transpose_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp?
$model_11/conv2d_transpose_12/BiasAddBiasAdd6model_11/conv2d_transpose_12/conv2d_transpose:output:0;model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2&
$model_11/conv2d_transpose_12/BiasAdd?
!model_11/conv2d_transpose_12/ReluRelu-model_11/conv2d_transpose_12/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2#
!model_11/conv2d_transpose_12/Relu?
"model_11/conv2d_transpose_13/ShapeShape/model_11/conv2d_transpose_12/Relu:activations:0*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_13/Shape?
0model_11/conv2d_transpose_13/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_11/conv2d_transpose_13/strided_slice/stack?
2model_11/conv2d_transpose_13/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_13/strided_slice/stack_1?
2model_11/conv2d_transpose_13/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_13/strided_slice/stack_2?
*model_11/conv2d_transpose_13/strided_sliceStridedSlice+model_11/conv2d_transpose_13/Shape:output:09model_11/conv2d_transpose_13/strided_slice/stack:output:0;model_11/conv2d_transpose_13/strided_slice/stack_1:output:0;model_11/conv2d_transpose_13/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_11/conv2d_transpose_13/strided_slice?
$model_11/conv2d_transpose_13/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_13/stack/1?
$model_11/conv2d_transpose_13/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_13/stack/2?
$model_11/conv2d_transpose_13/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_13/stack/3?
"model_11/conv2d_transpose_13/stackPack3model_11/conv2d_transpose_13/strided_slice:output:0-model_11/conv2d_transpose_13/stack/1:output:0-model_11/conv2d_transpose_13/stack/2:output:0-model_11/conv2d_transpose_13/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_13/stack?
2model_11/conv2d_transpose_13/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_11/conv2d_transpose_13/strided_slice_1/stack?
4model_11/conv2d_transpose_13/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_13/strided_slice_1/stack_1?
4model_11/conv2d_transpose_13/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_13/strided_slice_1/stack_2?
,model_11/conv2d_transpose_13/strided_slice_1StridedSlice+model_11/conv2d_transpose_13/stack:output:0;model_11/conv2d_transpose_13/strided_slice_1/stack:output:0=model_11/conv2d_transpose_13/strided_slice_1/stack_1:output:0=model_11/conv2d_transpose_13/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_11/conv2d_transpose_13/strided_slice_1?
<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_11_conv2d_transpose_13_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02>
<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp?
-model_11/conv2d_transpose_13/conv2d_transposeConv2DBackpropInput+model_11/conv2d_transpose_13/stack:output:0Dmodel_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp:value:0/model_11/conv2d_transpose_12/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-model_11/conv2d_transpose_13/conv2d_transpose?
3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOpReadVariableOp<model_11_conv2d_transpose_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp?
$model_11/conv2d_transpose_13/BiasAddBiasAdd6model_11/conv2d_transpose_13/conv2d_transpose:output:0;model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2&
$model_11/conv2d_transpose_13/BiasAdd?
!model_11/conv2d_transpose_13/ReluRelu-model_11/conv2d_transpose_13/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2#
!model_11/conv2d_transpose_13/Relu?
model_11/up_sampling2d_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2 
model_11/up_sampling2d_6/Const?
 model_11/up_sampling2d_6/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model_11/up_sampling2d_6/Const_1?
model_11/up_sampling2d_6/mulMul'model_11/up_sampling2d_6/Const:output:0)model_11/up_sampling2d_6/Const_1:output:0*
T0*
_output_shapes
:2
model_11/up_sampling2d_6/mul?
5model_11/up_sampling2d_6/resize/ResizeNearestNeighborResizeNearestNeighbor/model_11/conv2d_transpose_13/Relu:activations:0 model_11/up_sampling2d_6/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(27
5model_11/up_sampling2d_6/resize/ResizeNearestNeighbor?
"model_11/conv2d_transpose_14/ShapeShapeFmodel_11/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_14/Shape?
0model_11/conv2d_transpose_14/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_11/conv2d_transpose_14/strided_slice/stack?
2model_11/conv2d_transpose_14/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_14/strided_slice/stack_1?
2model_11/conv2d_transpose_14/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_14/strided_slice/stack_2?
*model_11/conv2d_transpose_14/strided_sliceStridedSlice+model_11/conv2d_transpose_14/Shape:output:09model_11/conv2d_transpose_14/strided_slice/stack:output:0;model_11/conv2d_transpose_14/strided_slice/stack_1:output:0;model_11/conv2d_transpose_14/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_11/conv2d_transpose_14/strided_slice?
$model_11/conv2d_transpose_14/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_14/stack/1?
$model_11/conv2d_transpose_14/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_14/stack/2?
$model_11/conv2d_transpose_14/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2&
$model_11/conv2d_transpose_14/stack/3?
"model_11/conv2d_transpose_14/stackPack3model_11/conv2d_transpose_14/strided_slice:output:0-model_11/conv2d_transpose_14/stack/1:output:0-model_11/conv2d_transpose_14/stack/2:output:0-model_11/conv2d_transpose_14/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_14/stack?
2model_11/conv2d_transpose_14/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_11/conv2d_transpose_14/strided_slice_1/stack?
4model_11/conv2d_transpose_14/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_14/strided_slice_1/stack_1?
4model_11/conv2d_transpose_14/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_14/strided_slice_1/stack_2?
,model_11/conv2d_transpose_14/strided_slice_1StridedSlice+model_11/conv2d_transpose_14/stack:output:0;model_11/conv2d_transpose_14/strided_slice_1/stack:output:0=model_11/conv2d_transpose_14/strided_slice_1/stack_1:output:0=model_11/conv2d_transpose_14/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_11/conv2d_transpose_14/strided_slice_1?
<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_11_conv2d_transpose_14_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02>
<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp?
-model_11/conv2d_transpose_14/conv2d_transposeConv2DBackpropInput+model_11/conv2d_transpose_14/stack:output:0Dmodel_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp:value:0Fmodel_11/up_sampling2d_6/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2/
-model_11/conv2d_transpose_14/conv2d_transpose?
3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOpReadVariableOp<model_11_conv2d_transpose_14_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp?
$model_11/conv2d_transpose_14/BiasAddBiasAdd6model_11/conv2d_transpose_14/conv2d_transpose:output:0;model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2&
$model_11/conv2d_transpose_14/BiasAdd?
!model_11/conv2d_transpose_14/ReluRelu-model_11/conv2d_transpose_14/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2#
!model_11/conv2d_transpose_14/Relu?
model_11/up_sampling2d_7/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2 
model_11/up_sampling2d_7/Const?
 model_11/up_sampling2d_7/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2"
 model_11/up_sampling2d_7/Const_1?
model_11/up_sampling2d_7/mulMul'model_11/up_sampling2d_7/Const:output:0)model_11/up_sampling2d_7/Const_1:output:0*
T0*
_output_shapes
:2
model_11/up_sampling2d_7/mul?
5model_11/up_sampling2d_7/resize/ResizeNearestNeighborResizeNearestNeighbor/model_11/conv2d_transpose_14/Relu:activations:0 model_11/up_sampling2d_7/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(27
5model_11/up_sampling2d_7/resize/ResizeNearestNeighbor?
"model_11/conv2d_transpose_15/ShapeShapeFmodel_11/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_15/Shape?
0model_11/conv2d_transpose_15/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_11/conv2d_transpose_15/strided_slice/stack?
2model_11/conv2d_transpose_15/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_15/strided_slice/stack_1?
2model_11/conv2d_transpose_15/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_11/conv2d_transpose_15/strided_slice/stack_2?
*model_11/conv2d_transpose_15/strided_sliceStridedSlice+model_11/conv2d_transpose_15/Shape:output:09model_11/conv2d_transpose_15/strided_slice/stack:output:0;model_11/conv2d_transpose_15/strided_slice/stack_1:output:0;model_11/conv2d_transpose_15/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_11/conv2d_transpose_15/strided_slice?
$model_11/conv2d_transpose_15/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_15/stack/1?
$model_11/conv2d_transpose_15/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_15/stack/2?
$model_11/conv2d_transpose_15/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2&
$model_11/conv2d_transpose_15/stack/3?
"model_11/conv2d_transpose_15/stackPack3model_11/conv2d_transpose_15/strided_slice:output:0-model_11/conv2d_transpose_15/stack/1:output:0-model_11/conv2d_transpose_15/stack/2:output:0-model_11/conv2d_transpose_15/stack/3:output:0*
N*
T0*
_output_shapes
:2$
"model_11/conv2d_transpose_15/stack?
2model_11/conv2d_transpose_15/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_11/conv2d_transpose_15/strided_slice_1/stack?
4model_11/conv2d_transpose_15/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_15/strided_slice_1/stack_1?
4model_11/conv2d_transpose_15/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_11/conv2d_transpose_15/strided_slice_1/stack_2?
,model_11/conv2d_transpose_15/strided_slice_1StridedSlice+model_11/conv2d_transpose_15/stack:output:0;model_11/conv2d_transpose_15/strided_slice_1/stack:output:0=model_11/conv2d_transpose_15/strided_slice_1/stack_1:output:0=model_11/conv2d_transpose_15/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_11/conv2d_transpose_15/strided_slice_1?
<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOpReadVariableOpEmodel_11_conv2d_transpose_15_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02>
<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp?
-model_11/conv2d_transpose_15/conv2d_transposeConv2DBackpropInput+model_11/conv2d_transpose_15/stack:output:0Dmodel_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:value:0Fmodel_11/up_sampling2d_7/resize/ResizeNearestNeighbor:resized_images:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2/
-model_11/conv2d_transpose_15/conv2d_transpose?
3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOpReadVariableOp<model_11_conv2d_transpose_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp?
$model_11/conv2d_transpose_15/BiasAddBiasAdd6model_11/conv2d_transpose_15/conv2d_transpose:output:0;model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2&
$model_11/conv2d_transpose_15/BiasAdd?
$model_11/conv2d_transpose_15/SigmoidSigmoid-model_11/conv2d_transpose_15/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2&
$model_11/conv2d_transpose_15/Sigmoid?
IdentityIdentity(model_11/conv2d_transpose_15/Sigmoid:y:0*^model_10/conv2d_10/BiasAdd/ReadVariableOp)^model_10/conv2d_10/Conv2D/ReadVariableOp*^model_10/conv2d_11/BiasAdd/ReadVariableOp)^model_10/conv2d_11/Conv2D/ReadVariableOp)^model_10/conv2d_8/BiasAdd/ReadVariableOp(^model_10/conv2d_8/Conv2D/ReadVariableOp)^model_10/conv2d_9/BiasAdd/ReadVariableOp(^model_10/conv2d_9/Conv2D/ReadVariableOp4^model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp=^model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp4^model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp=^model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp4^model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp=^model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp4^model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp=^model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 2V
)model_10/conv2d_10/BiasAdd/ReadVariableOp)model_10/conv2d_10/BiasAdd/ReadVariableOp2T
(model_10/conv2d_10/Conv2D/ReadVariableOp(model_10/conv2d_10/Conv2D/ReadVariableOp2V
)model_10/conv2d_11/BiasAdd/ReadVariableOp)model_10/conv2d_11/BiasAdd/ReadVariableOp2T
(model_10/conv2d_11/Conv2D/ReadVariableOp(model_10/conv2d_11/Conv2D/ReadVariableOp2T
(model_10/conv2d_8/BiasAdd/ReadVariableOp(model_10/conv2d_8/BiasAdd/ReadVariableOp2R
'model_10/conv2d_8/Conv2D/ReadVariableOp'model_10/conv2d_8/Conv2D/ReadVariableOp2T
(model_10/conv2d_9/BiasAdd/ReadVariableOp(model_10/conv2d_9/BiasAdd/ReadVariableOp2R
'model_10/conv2d_9/Conv2D/ReadVariableOp'model_10/conv2d_9/Conv2D/ReadVariableOp2j
3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp3model_11/conv2d_transpose_12/BiasAdd/ReadVariableOp2|
<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp<model_11/conv2d_transpose_12/conv2d_transpose/ReadVariableOp2j
3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp3model_11/conv2d_transpose_13/BiasAdd/ReadVariableOp2|
<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp<model_11/conv2d_transpose_13/conv2d_transpose/ReadVariableOp2j
3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp3model_11/conv2d_transpose_14/BiasAdd/ReadVariableOp2|
<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp<model_11/conv2d_transpose_14/conv2d_transpose/ReadVariableOp2j
3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp3model_11/conv2d_transpose_15/BiasAdd/ReadVariableOp2|
<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp<model_11/conv2d_transpose_15/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_model_12_layer_call_fn_21077

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11: 

unknown_12: $

unknown_13: 

unknown_14:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_12_layer_call_and_return_conditional_losses_207352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*N
_input_shapes=
;:?????????: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
(__inference_model_10_layer_call_fn_21403

inputs!
unknown: 
	unknown_0: #
	unknown_1: 
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_10_layer_call_and_return_conditional_losses_200512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_14_layer_call_fn_20414

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_204042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?%
?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_20295

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_21802

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?%
?
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_20468

inputsB
(conv2d_transpose_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
`
D__inference_flatten_2_layer_call_and_return_conditional_losses_21833

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_21822

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_58
serving_default_input_5:0?????????D
model_118
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
̜
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api
	
signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"name": "model_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["flatten_2", 0, 0]]}, "name": "model_10", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 196]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 4]}}, "name": "reshape_3", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_12", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_12", "inbound_nodes": [[["reshape_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_13", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_13", "inbound_nodes": [[["conv2d_transpose_12", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_6", "inbound_nodes": [[["conv2d_transpose_13", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_14", "inbound_nodes": [[["up_sampling2d_6", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_7", "inbound_nodes": [[["conv2d_transpose_14", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_15", "inbound_nodes": [[["up_sampling2d_7", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["conv2d_transpose_15", 0, 0]]}, "name": "model_11", "inbound_nodes": [[["model_10", 1, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["model_11", 1, 0]]}, "shared_object_id": 34, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "float32", "input_5"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_12", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["flatten_2", 0, 0]]}, "name": "model_10", "inbound_nodes": [[["input_5", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Functional", "config": {"name": "model_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 196]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 4]}}, "name": "reshape_3", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_12", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_12", "inbound_nodes": [[["reshape_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_13", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_13", "inbound_nodes": [[["conv2d_transpose_12", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_6", "inbound_nodes": [[["conv2d_transpose_13", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_14", "inbound_nodes": [[["up_sampling2d_6", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_7", "inbound_nodes": [[["conv2d_transpose_14", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_15", "inbound_nodes": [[["up_sampling2d_7", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["conv2d_transpose_15", 0, 0]]}, "name": "model_11", "inbound_nodes": [[["model_10", 1, 0, {}]]], "shared_object_id": 33}], "input_layers": [["input_5", 0, 0]], "output_layers": [["model_11", 1, 0]]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}, "shared_object_id": 36}, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0020000000949949026, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
?N
layer-0

layer_with_weights-0

layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	variables
regularization_losses
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?K
_tf_keras_network?K{"name": "model_10", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["flatten_2", 0, 0]]}, "inbound_nodes": [[["input_5", 0, 0, {}]]], "shared_object_id": 16, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28, 28, 1]}, "float32", "input_5"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_10", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28, 28, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["input_5", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_4", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_9", "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_5", "inbound_nodes": [[["conv2d_9", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_11", "inbound_nodes": [[["conv2d_10", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_2", "inbound_nodes": [[["conv2d_11", 0, 0, {}]]], "shared_object_id": 15}], "input_layers": [["input_5", 0, 0]], "output_layers": [["flatten_2", 0, 0]]}}}
?P
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	variables
regularization_losses
trainable_variables
 	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?N
_tf_keras_network?M{"name": "model_11", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 196]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 4]}}, "name": "reshape_3", "inbound_nodes": [[["input_6", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_12", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_12", "inbound_nodes": [[["reshape_3", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_13", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_13", "inbound_nodes": [[["conv2d_transpose_12", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_6", "inbound_nodes": [[["conv2d_transpose_13", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_14", "inbound_nodes": [[["up_sampling2d_6", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_7", "inbound_nodes": [[["conv2d_transpose_14", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_15", "inbound_nodes": [[["up_sampling2d_7", 0, 0, {}]]]}], "input_layers": [["input_6", 0, 0]], "output_layers": [["conv2d_transpose_15", 0, 0]]}, "inbound_nodes": [[["model_10", 1, 0, {}]]], "shared_object_id": 33, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 196]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 196]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 196]}, "float32", "input_6"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_11", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 196]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}, "name": "input_6", "inbound_nodes": [], "shared_object_id": 17}, {"class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 4]}}, "name": "reshape_3", "inbound_nodes": [[["input_6", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_12", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_12", "inbound_nodes": [[["reshape_3", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_13", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_13", "inbound_nodes": [[["conv2d_transpose_12", 0, 0, {}]]], "shared_object_id": 24}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_6", "inbound_nodes": [[["conv2d_transpose_13", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_14", "inbound_nodes": [[["up_sampling2d_6", 0, 0, {}]]], "shared_object_id": 28}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_7", "inbound_nodes": [[["conv2d_transpose_14", 0, 0, {}]]], "shared_object_id": 29}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_15", "inbound_nodes": [[["up_sampling2d_7", 0, 0, {}]]], "shared_object_id": 32}], "input_layers": [["input_6", 0, 0]], "output_layers": [["conv2d_transpose_15", 0, 0]]}}}
?
!iter

"beta_1

#beta_2
	$decay
%learning_rate&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?2m?3m?4m?5m?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?2v?3v?4v?5v?"
	optimizer
?
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
414
515"
trackable_list_wrapper
 "
trackable_list_wrapper
?
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
212
313
414
515"
trackable_list_wrapper
?
6metrics

7layers
	variables
8non_trainable_variables
9layer_metrics
regularization_losses
trainable_variables
:layer_regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?


&kernel
'bias
;	variables
<regularization_losses
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_5", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?
?	variables
@regularization_losses
Atrainable_variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_4", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 40}}
?

(kernel
)bias
C	variables
Dregularization_losses
Etrainable_variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_9", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["max_pooling2d_4", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 32]}}
?
G	variables
Hregularization_losses
Itrainable_variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_5", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_9", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 42}}
?

*kernel
+bias
K	variables
Lregularization_losses
Mtrainable_variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["max_pooling2d_5", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 16]}}
?


,kernel
-bias
O	variables
Pregularization_losses
Qtrainable_variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_11", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["conv2d_10", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 8]}}
?
S	variables
Tregularization_losses
Utrainable_variables
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_2", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["conv2d_11", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 45}}
X
&0
'1
(2
)3
*4
+5
,6
-7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
&0
'1
(2
)3
*4
+5
,6
-7"
trackable_list_wrapper
?
Wmetrics

Xlayers
	variables
Ynon_trainable_variables
Zlayer_metrics
regularization_losses
trainable_variables
[layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_6", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 196]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 196]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_6"}}
?
\	variables
]regularization_losses
^trainable_variables
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "reshape_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_3", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 4]}}, "inbound_nodes": [[["input_6", 0, 0, {}]]], "shared_object_id": 18}
?

.kernel
/bias
`	variables
aregularization_losses
btrainable_variables
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "conv2d_transpose_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_12", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["reshape_3", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 4}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 4]}}
?

0kernel
1bias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "conv2d_transpose_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_13", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["conv2d_transpose_12", 0, 0, {}]]], "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 8}}, "shared_object_id": 47}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 8]}}
?
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up_sampling2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_6", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "inbound_nodes": [[["conv2d_transpose_13", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 48}}
?

2kernel
3bias
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "conv2d_transpose_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_14", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["up_sampling2d_6", 0, 0, {}]]], "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 16}}, "shared_object_id": 49}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 16]}}
?
p	variables
qregularization_losses
rtrainable_variables
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "up_sampling2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_7", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "inbound_nodes": [[["conv2d_transpose_14", 0, 0, {}]]], "shared_object_id": 29, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 50}}
?

4kernel
5bias
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "conv2d_transpose_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_15", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 30}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 31}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "inbound_nodes": [[["up_sampling2d_7", 0, 0, {}]]], "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}, "shared_object_id": 51}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
X
.0
/1
02
13
24
35
46
57"
trackable_list_wrapper
 "
trackable_list_wrapper
X
.0
/1
02
13
24
35
46
57"
trackable_list_wrapper
?
xmetrics

ylayers
	variables
znon_trainable_variables
{layer_metrics
regularization_losses
trainable_variables
|layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):' 2conv2d_8/kernel
: 2conv2d_8/bias
):' 2conv2d_9/kernel
:2conv2d_9/bias
*:(2conv2d_10/kernel
:2conv2d_10/bias
*:(2conv2d_11/kernel
:2conv2d_11/bias
4:22conv2d_transpose_12/kernel
&:$2conv2d_transpose_12/bias
4:22conv2d_transpose_13/kernel
&:$2conv2d_transpose_13/bias
4:2 2conv2d_transpose_14/kernel
&:$ 2conv2d_transpose_14/bias
4:2 2conv2d_transpose_15/kernel
&:$2conv2d_transpose_15/bias
'
}0"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?
~metrics

layers
;	variables
?non_trainable_variables
?layer_metrics
<regularization_losses
=trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
?	variables
?non_trainable_variables
?layer_metrics
@regularization_losses
Atrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
?metrics
?layers
C	variables
?non_trainable_variables
?layer_metrics
Dregularization_losses
Etrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
G	variables
?non_trainable_variables
?layer_metrics
Hregularization_losses
Itrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
?metrics
?layers
K	variables
?non_trainable_variables
?layer_metrics
Lregularization_losses
Mtrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
?metrics
?layers
O	variables
?non_trainable_variables
?layer_metrics
Pregularization_losses
Qtrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
S	variables
?non_trainable_variables
?layer_metrics
Tregularization_losses
Utrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
X
0

1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
\	variables
?non_trainable_variables
?layer_metrics
]regularization_losses
^trainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
?metrics
?layers
`	variables
?non_trainable_variables
?layer_metrics
aregularization_losses
btrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
?metrics
?layers
d	variables
?non_trainable_variables
?layer_metrics
eregularization_losses
ftrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
h	variables
?non_trainable_variables
?layer_metrics
iregularization_losses
jtrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
?
?metrics
?layers
l	variables
?non_trainable_variables
?layer_metrics
mregularization_losses
ntrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
?layers
p	variables
?non_trainable_variables
?layer_metrics
qregularization_losses
rtrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
?metrics
?layers
t	variables
?non_trainable_variables
?layer_metrics
uregularization_losses
vtrainable_variables
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 52}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
.:, 2Adam/conv2d_8/kernel/m
 : 2Adam/conv2d_8/bias/m
.:, 2Adam/conv2d_9/kernel/m
 :2Adam/conv2d_9/bias/m
/:-2Adam/conv2d_10/kernel/m
!:2Adam/conv2d_10/bias/m
/:-2Adam/conv2d_11/kernel/m
!:2Adam/conv2d_11/bias/m
9:72!Adam/conv2d_transpose_12/kernel/m
+:)2Adam/conv2d_transpose_12/bias/m
9:72!Adam/conv2d_transpose_13/kernel/m
+:)2Adam/conv2d_transpose_13/bias/m
9:7 2!Adam/conv2d_transpose_14/kernel/m
+:) 2Adam/conv2d_transpose_14/bias/m
9:7 2!Adam/conv2d_transpose_15/kernel/m
+:)2Adam/conv2d_transpose_15/bias/m
.:, 2Adam/conv2d_8/kernel/v
 : 2Adam/conv2d_8/bias/v
.:, 2Adam/conv2d_9/kernel/v
 :2Adam/conv2d_9/bias/v
/:-2Adam/conv2d_10/kernel/v
!:2Adam/conv2d_10/bias/v
/:-2Adam/conv2d_11/kernel/v
!:2Adam/conv2d_11/bias/v
9:72!Adam/conv2d_transpose_12/kernel/v
+:)2Adam/conv2d_transpose_12/bias/v
9:72!Adam/conv2d_transpose_13/kernel/v
+:)2Adam/conv2d_transpose_13/bias/v
9:7 2!Adam/conv2d_transpose_14/kernel/v
+:) 2Adam/conv2d_transpose_14/bias/v
9:7 2!Adam/conv2d_transpose_15/kernel/v
+:)2Adam/conv2d_transpose_15/bias/v
?2?
(__inference_model_12_layer_call_fn_20770
(__inference_model_12_layer_call_fn_21077
(__inference_model_12_layer_call_fn_21114
(__inference_model_12_layer_call_fn_20919?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_19941?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_5?????????
?2?
C__inference_model_12_layer_call_and_return_conditional_losses_21248
C__inference_model_12_layer_call_and_return_conditional_losses_21382
C__inference_model_12_layer_call_and_return_conditional_losses_20957
C__inference_model_12_layer_call_and_return_conditional_losses_20995?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_10_layer_call_fn_20070
(__inference_model_10_layer_call_fn_21403
(__inference_model_10_layer_call_fn_21424
(__inference_model_10_layer_call_fn_20206?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_10_layer_call_and_return_conditional_losses_21460
C__inference_model_10_layer_call_and_return_conditional_losses_21496
C__inference_model_10_layer_call_and_return_conditional_losses_20233
C__inference_model_10_layer_call_and_return_conditional_losses_20260?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_model_11_layer_call_fn_20543
(__inference_model_11_layer_call_fn_21517
(__inference_model_11_layer_call_fn_21538
(__inference_model_11_layer_call_fn_20639?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_11_layer_call_and_return_conditional_losses_21640
C__inference_model_11_layer_call_and_return_conditional_losses_21742
C__inference_model_11_layer_call_and_return_conditional_losses_20666
C__inference_model_11_layer_call_and_return_conditional_losses_20693?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
#__inference_signature_wrapper_21040input_5"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_8_layer_call_fn_21751?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_8_layer_call_and_return_conditional_losses_21762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_4_layer_call_fn_19953?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_19947?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
(__inference_conv2d_9_layer_call_fn_21771?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_21782?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_5_layer_call_fn_19965?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_19959?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_conv2d_10_layer_call_fn_21791?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_10_layer_call_and_return_conditional_losses_21802?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_11_layer_call_fn_21811?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_11_layer_call_and_return_conditional_losses_21822?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_2_layer_call_fn_21827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_2_layer_call_and_return_conditional_losses_21833?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_3_layer_call_fn_21838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_3_layer_call_and_return_conditional_losses_21852?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
3__inference_conv2d_transpose_12_layer_call_fn_20305?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_20295?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
3__inference_conv2d_transpose_13_layer_call_fn_20350?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_20340?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
/__inference_up_sampling2d_6_layer_call_fn_20369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_20363?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
3__inference_conv2d_transpose_14_layer_call_fn_20414?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_20404?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????
?2?
/__inference_up_sampling2d_7_layer_call_fn_20433?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_20427?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
3__inference_conv2d_transpose_15_layer_call_fn_20478?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_20468?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? ?
 __inference__wrapped_model_19941?&'()*+,-./0123458?5
.?+
)?&
input_5?????????
? ";?8
6
model_11*?'
model_11??????????
D__inference_conv2d_10_layer_call_and_return_conditional_losses_21802l*+7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_10_layer_call_fn_21791_*+7?4
-?*
(?%
inputs?????????
? " ???????????
D__inference_conv2d_11_layer_call_and_return_conditional_losses_21822l,-7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_conv2d_11_layer_call_fn_21811_,-7?4
-?*
(?%
inputs?????????
? " ???????????
C__inference_conv2d_8_layer_call_and_return_conditional_losses_21762l&'7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
(__inference_conv2d_8_layer_call_fn_21751_&'7?4
-?*
(?%
inputs?????????
? " ?????????? ?
C__inference_conv2d_9_layer_call_and_return_conditional_losses_21782l()7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????
? ?
(__inference_conv2d_9_layer_call_fn_21771_()7?4
-?*
(?%
inputs????????? 
? " ???????????
N__inference_conv2d_transpose_12_layer_call_and_return_conditional_losses_20295?./I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_12_layer_call_fn_20305?./I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
N__inference_conv2d_transpose_13_layer_call_and_return_conditional_losses_20340?01I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_13_layer_call_fn_20350?01I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
N__inference_conv2d_transpose_14_layer_call_and_return_conditional_losses_20404?23I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+??????????????????????????? 
? ?
3__inference_conv2d_transpose_14_layer_call_fn_20414?23I?F
??<
:?7
inputs+???????????????????????????
? "2?/+??????????????????????????? ?
N__inference_conv2d_transpose_15_layer_call_and_return_conditional_losses_20468?45I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
3__inference_conv2d_transpose_15_layer_call_fn_20478?45I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
D__inference_flatten_2_layer_call_and_return_conditional_losses_21833a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten_2_layer_call_fn_21827T7?4
-?*
(?%
inputs?????????
? "????????????
J__inference_max_pooling2d_4_layer_call_and_return_conditional_losses_19947?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_4_layer_call_fn_19953?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_5_layer_call_and_return_conditional_losses_19959?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_5_layer_call_fn_19965?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
C__inference_model_10_layer_call_and_return_conditional_losses_20233t&'()*+,-@?=
6?3
)?&
input_5?????????
p 

 
? "&?#
?
0??????????
? ?
C__inference_model_10_layer_call_and_return_conditional_losses_20260t&'()*+,-@?=
6?3
)?&
input_5?????????
p

 
? "&?#
?
0??????????
? ?
C__inference_model_10_layer_call_and_return_conditional_losses_21460s&'()*+,-??<
5?2
(?%
inputs?????????
p 

 
? "&?#
?
0??????????
? ?
C__inference_model_10_layer_call_and_return_conditional_losses_21496s&'()*+,-??<
5?2
(?%
inputs?????????
p

 
? "&?#
?
0??????????
? ?
(__inference_model_10_layer_call_fn_20070g&'()*+,-@?=
6?3
)?&
input_5?????????
p 

 
? "????????????
(__inference_model_10_layer_call_fn_20206g&'()*+,-@?=
6?3
)?&
input_5?????????
p

 
? "????????????
(__inference_model_10_layer_call_fn_21403f&'()*+,-??<
5?2
(?%
inputs?????????
p 

 
? "????????????
(__inference_model_10_layer_call_fn_21424f&'()*+,-??<
5?2
(?%
inputs?????????
p

 
? "????????????
C__inference_model_11_layer_call_and_return_conditional_losses_20666?./0123459?6
/?,
"?
input_6??????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_11_layer_call_and_return_conditional_losses_20693?./0123459?6
/?,
"?
input_6??????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_11_layer_call_and_return_conditional_losses_21640s./0123458?5
.?+
!?
inputs??????????
p 

 
? "-?*
#? 
0?????????
? ?
C__inference_model_11_layer_call_and_return_conditional_losses_21742s./0123458?5
.?+
!?
inputs??????????
p

 
? "-?*
#? 
0?????????
? ?
(__inference_model_11_layer_call_fn_20543y./0123459?6
/?,
"?
input_6??????????
p 

 
? "2?/+????????????????????????????
(__inference_model_11_layer_call_fn_20639y./0123459?6
/?,
"?
input_6??????????
p

 
? "2?/+????????????????????????????
(__inference_model_11_layer_call_fn_21517x./0123458?5
.?+
!?
inputs??????????
p 

 
? "2?/+????????????????????????????
(__inference_model_11_layer_call_fn_21538x./0123458?5
.?+
!?
inputs??????????
p

 
? "2?/+????????????????????????????
C__inference_model_12_layer_call_and_return_conditional_losses_20957?&'()*+,-./012345@?=
6?3
)?&
input_5?????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_12_layer_call_and_return_conditional_losses_20995?&'()*+,-./012345@?=
6?3
)?&
input_5?????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_12_layer_call_and_return_conditional_losses_21248?&'()*+,-./012345??<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????
? ?
C__inference_model_12_layer_call_and_return_conditional_losses_21382?&'()*+,-./012345??<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????
? ?
(__inference_model_12_layer_call_fn_20770?&'()*+,-./012345@?=
6?3
)?&
input_5?????????
p 

 
? "2?/+????????????????????????????
(__inference_model_12_layer_call_fn_20919?&'()*+,-./012345@?=
6?3
)?&
input_5?????????
p

 
? "2?/+????????????????????????????
(__inference_model_12_layer_call_fn_21077?&'()*+,-./012345??<
5?2
(?%
inputs?????????
p 

 
? "2?/+????????????????????????????
(__inference_model_12_layer_call_fn_21114?&'()*+,-./012345??<
5?2
(?%
inputs?????????
p

 
? "2?/+????????????????????????????
D__inference_reshape_3_layer_call_and_return_conditional_losses_21852a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_3_layer_call_fn_21838T0?-
&?#
!?
inputs??????????
? " ???????????
#__inference_signature_wrapper_21040?&'()*+,-./012345C?@
? 
9?6
4
input_5)?&
input_5?????????";?8
6
model_11*?'
model_11??????????
J__inference_up_sampling2d_6_layer_call_and_return_conditional_losses_20363?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_6_layer_call_fn_20369?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_7_layer_call_and_return_conditional_losses_20427?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_7_layer_call_fn_20433?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????