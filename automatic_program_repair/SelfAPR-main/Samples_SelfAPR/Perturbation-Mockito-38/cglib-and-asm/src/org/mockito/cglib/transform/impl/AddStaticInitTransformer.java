[P8_Replace_Mix]^info =  ReflectUtils.getMethodInfo ( null ) ;^31^^^^^30^41^info = ReflectUtils.getMethodInfo ( classInit ) ;^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P14_Delete_Statement]^^31^^^^^30^41^info = ReflectUtils.getMethodInfo ( classInit ) ;^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^if  ( !TypeUtils.isInterface ( info.getModifiers (  )  )  )  {^32^^^^^30^41^if  ( !TypeUtils.isStatic ( info.getModifiers (  )  )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^if  ( !TypeUtils.isStatic ( info.getSignature (  )  )  )  {^32^^^^^30^41^if  ( !TypeUtils.isStatic ( info.getModifiers (  )  )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P9_Replace_Statement]^if  ( !TypeUtils.isInterface ( getAccess (  )  )  )  {^32^^^^^30^41^if  ( !TypeUtils.isStatic ( info.getModifiers (  )  )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException((classInit + " is not static"));^32^33^34^^^30^41^if  ( !TypeUtils.isStatic ( info.getModifiers (  )  )  )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P16_Remove_Block]^^32^33^34^^^30^41^if  ( !TypeUtils.isStatic ( info.getModifiers (  )  )  )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P4_Replace_Constructor]^throw throw  new IllegalArgumentException (  ( classInit + " illegal signature" )  )   ;^33^^^^^30^41^throw new IllegalArgumentException  (" ")  ;^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^Type[] types = info.getSignature (  ) .getReturnType (  ) ;^35^^^^^30^41^Type[] types = info.getSignature (  ) .getArgumentTypes (  ) ;^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^Type[] types = info.getModifiers (  ) .getArgumentTypes (  ) ;^35^^^^^30^41^Type[] types = info.getSignature (  ) .getArgumentTypes (  ) ;^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P14_Delete_Statement]^^35^^^^^30^41^Type[] types = info.getSignature (  ) .getArgumentTypes (  ) ;^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P2_Replace_Operator]^if  ( types.length != 1 && !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P2_Replace_Operator]^if  ( types.length >= 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P3_Replace_Literal]^if  ( types.length != 0 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P3_Replace_Literal]^if  ( types.length != 1 || !types[-1].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P6_Replace_Expression]^if  ( types.length != 1 ) {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P6_Replace_Expression]^if  (  !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P6_Replace_Expression]^if  ( (types.length != 1 || TYPE_CLASS))) )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P6_Replace_Expression]^if  ( classInit + " illegal signature" )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^if  ( types.length != 1 || !types[0] .getAccess (  )   || !info.getSignature (  ) .getReturnType (  )^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^if  ( types.length != 1 || !types[0].getAccess ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getArgumentTypes (  ) .equals ( Type.VOID_TYPE )  )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getModifiers (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P8_Replace_Mix]^if  (  !types[0].equals ( Constants.TYPE_CLASS )  || !info.getModifiers (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^36^37^38^^^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P15_Unwrap_Block]^throw new java.lang.IllegalArgumentException((classInit + " illegal signature"));^36^37^38^39^40^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P16_Remove_Block]^^36^37^38^39^40^30^41^if  ( types.length != 1 || !types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  { throw new IllegalArgumentException  (" ")  ; }^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P4_Replace_Constructor]^throw throw  new IllegalArgumentException (  ( classInit + " is not static" )  )   ;^39^^^^^30^41^throw new IllegalArgumentException  (" ")  ;^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P3_Replace_Literal]^!types[8].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^37^38^^^^30^41^!types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P14_Delete_Statement]^^37^38^39^^^30^41^!types[0].equals ( Constants.TYPE_CLASS )  || !info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  { throw new IllegalArgumentException  (" ")  ;^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^!info.getSignature (  ) .getReturnType (  ) .getAccess ( Type.VOID_TYPE )  )  {^38^^^^^30^41^!info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^!info.getSignature (  ) .getArgumentTypes (  ) .equals ( Type.VOID_TYPE )  )  {^38^^^^^30^41^!info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^!info.getModifiers (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^38^^^^^30^41^!info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P14_Delete_Statement]^^38^39^^^^30^41^!info.getSignature (  ) .getReturnType (  ) .equals ( Type.VOID_TYPE )  )  { throw new IllegalArgumentException  (" ")  ;^[CLASS] AddStaticInitTransformer  [METHOD] <init> [RETURN_TYPE] Method)   Method classInit [VARIABLES] Type[]  types  MethodInfo  info  Method  classInit  boolean  
[P7_Replace_Invocation]^if  ( !TypeUtils.isStatic ( getAccess (  )  )  )  {^44^^^^^43^49^if  ( !TypeUtils.isInterface ( getAccess (  )  )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P7_Replace_Invocation]^if  ( !TypeUtils.isInterface ( getStaticHook (  )  )  )  {^44^^^^^43^49^if  ( !TypeUtils.isInterface ( getAccess (  )  )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P9_Replace_Statement]^if  ( !TypeUtils.isStatic ( info.getModifiers (  )  )  )  {^44^^^^^43^49^if  ( !TypeUtils.isInterface ( getAccess (  )  )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P15_Unwrap_Block]^org.mockito.cglib.transform.impl.CodeEmitter e = getStaticHook(); org.mockito.cglib.transform.impl.EmitUtils.load_class_this(e); e.invoke(info);^44^45^46^47^48^43^49^if  ( !TypeUtils.isInterface ( getAccess (  )  )  )  { CodeEmitter e = getStaticHook (  ) ; EmitUtils.load_class_this ( e ) ; e.invoke ( info ) ; }^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P16_Remove_Block]^^44^45^46^47^48^43^49^if  ( !TypeUtils.isInterface ( getAccess (  )  )  )  { CodeEmitter e = getStaticHook (  ) ; EmitUtils.load_class_this ( e ) ; e.invoke ( info ) ; }^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P7_Replace_Invocation]^CodeEmitter e = getAccess (  ) ;^45^^^^^43^49^CodeEmitter e = getStaticHook (  ) ;^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P14_Delete_Statement]^^45^46^^^^43^49^CodeEmitter e = getStaticHook (  ) ; EmitUtils.load_class_this ( e ) ;^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P14_Delete_Statement]^^46^47^^^^43^49^EmitUtils.load_class_this ( e ) ; e.invoke ( info ) ;^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P14_Delete_Statement]^^47^^^^^43^49^e.invoke ( info ) ;^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P7_Replace_Invocation]^if  ( !TypeUtils .isStatic (  )   )  {^44^^^^^43^49^if  ( !TypeUtils.isInterface ( getAccess (  )  )  )  {^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
[P14_Delete_Statement]^^45^^^^^43^49^CodeEmitter e = getStaticHook (  ) ;^[CLASS] AddStaticInitTransformer  [METHOD] init [RETURN_TYPE] void   [VARIABLES] CodeEmitter  e  MethodInfo  info  boolean  
