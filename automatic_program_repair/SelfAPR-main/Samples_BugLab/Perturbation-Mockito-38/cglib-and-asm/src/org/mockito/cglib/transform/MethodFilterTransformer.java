[BugLab_Variable_Misuse]^return  ( filter.accept ( access, signature, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Variable_Misuse]^return  ( filter.accept ( access, name, signature, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Variable_Misuse]^return  ( filter.accept ( access, name, desc, desc, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( pass, name, desc, signature, exceptions )  ? access : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( access, desc, name, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( access, signature, desc, name, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( access, name, desc, signature, direct )  ? pass : exceptions ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( exceptions.accept ( access, name, desc, signature, filter )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( access, direct, desc, signature, exceptions )  ? pass : name ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Variable_Misuse]^return  ( filter.accept ( access, name, desc, name, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( desc, name, access, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( access, exceptions, desc, signature, name )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( access, name, signature, desc, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( access, name, exceptions, signature, desc )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( access, name, pass, signature, exceptions )  ? desc : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Argument_Swapping]^return  ( filter.accept ( access, name, desc, direct, exceptions )  ? pass : signature ) .visitMethod ( access, name, desc, signature, exceptions ) ;^36^^^^^31^37^return  ( filter.accept ( access, name, desc, signature, exceptions )  ? pass : direct ) .visitMethod ( access, name, desc, signature, exceptions ) ;^[CLASS] MethodFilterTransformer  [METHOD] visitMethod [RETURN_TYPE] MethodVisitor   int access String name String desc String signature String[] exceptions [VARIABLES] boolean  MethodFilter  filter  ClassTransformer  pass  ClassVisitor  direct  String  desc  name  signature  String[]  exceptions  int  access  
[BugLab_Variable_Misuse]^pass.setTarget ( direct ) ;^40^^^^^39^42^pass.setTarget ( target ) ;^[CLASS] MethodFilterTransformer  [METHOD] setTarget [RETURN_TYPE] void   ClassVisitor target [VARIABLES] ClassTransformer  pass  ClassVisitor  direct  target  boolean  MethodFilter  filter  
[BugLab_Variable_Misuse]^direct = direct;^41^^^^^39^42^direct = target;^[CLASS] MethodFilterTransformer  [METHOD] setTarget [RETURN_TYPE] void   ClassVisitor target [VARIABLES] ClassTransformer  pass  ClassVisitor  direct  target  boolean  MethodFilter  filter  