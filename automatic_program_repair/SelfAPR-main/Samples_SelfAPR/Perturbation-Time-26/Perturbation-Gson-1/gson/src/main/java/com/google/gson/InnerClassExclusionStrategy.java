[P7_Replace_Invocation]^return isStatic ( f.getDeclaredClass (  )  ) ;^29^^^^^28^30^return isInnerClass ( f.getDeclaredClass (  )  ) ;^[CLASS] InnerClassExclusionStrategy  [METHOD] shouldSkipField [RETURN_TYPE] boolean   FieldAttributes f [VARIABLES] boolean  FieldAttributes  f  
[P14_Delete_Statement]^^29^^^^^28^30^return isInnerClass ( f.getDeclaredClass (  )  ) ;^[CLASS] InnerClassExclusionStrategy  [METHOD] shouldSkipField [RETURN_TYPE] boolean   FieldAttributes f [VARIABLES] boolean  FieldAttributes  f  
[P7_Replace_Invocation]^return isStatic ( clazz ) ;^33^^^^^32^34^return isInnerClass ( clazz ) ;^[CLASS] InnerClassExclusionStrategy  [METHOD] shouldSkipClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P14_Delete_Statement]^^33^^^^^32^34^return isInnerClass ( clazz ) ;^[CLASS] InnerClassExclusionStrategy  [METHOD] shouldSkipClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P2_Replace_Operator]^return clazz.isMemberClass (  )  || !isStatic ( clazz ) ;^37^^^^^36^38^return clazz.isMemberClass (  )  && !isStatic ( clazz ) ;^[CLASS] InnerClassExclusionStrategy  [METHOD] isInnerClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P7_Replace_Invocation]^return clazz.isMemberClass (  )  && !isInnerClass ( clazz ) ;^37^^^^^36^38^return clazz.isMemberClass (  )  && !isStatic ( clazz ) ;^[CLASS] InnerClassExclusionStrategy  [METHOD] isInnerClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P8_Replace_Mix]^return clazz .getModifiers (  )   && !isStatic ( clazz ) ;^37^^^^^36^38^return clazz.isMemberClass (  )  && !isStatic ( clazz ) ;^[CLASS] InnerClassExclusionStrategy  [METHOD] isInnerClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P14_Delete_Statement]^^37^^^^^36^38^return clazz.isMemberClass (  )  && !isStatic ( clazz ) ;^[CLASS] InnerClassExclusionStrategy  [METHOD] isInnerClass [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P2_Replace_Operator]^return  ( clazz.getModifiers (  )  & Modifier.STATIC )  <= 0;^41^^^^^40^42^return  ( clazz.getModifiers (  )  & Modifier.STATIC )  != 0;^[CLASS] InnerClassExclusionStrategy  [METHOD] isStatic [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P2_Replace_Operator]^return  ( clazz.getModifiers (  )   ^  Modifier.STATIC )  != 0;^41^^^^^40^42^return  ( clazz.getModifiers (  )  & Modifier.STATIC )  != 0;^[CLASS] InnerClassExclusionStrategy  [METHOD] isStatic [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P3_Replace_Literal]^return  ( clazz.getModifiers (  )  & Modifier.STATIC )  != 2;^41^^^^^40^42^return  ( clazz.getModifiers (  )  & Modifier.STATIC )  != 0;^[CLASS] InnerClassExclusionStrategy  [METHOD] isStatic [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P7_Replace_Invocation]^return  ( clazz .isMemberClass (  )   & Modifier.STATIC )  != 0;^41^^^^^40^42^return  ( clazz.getModifiers (  )  & Modifier.STATIC )  != 0;^[CLASS] InnerClassExclusionStrategy  [METHOD] isStatic [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
[P14_Delete_Statement]^^41^42^^^^40^42^return  ( clazz.getModifiers (  )  & Modifier.STATIC )  != 0; }^[CLASS] InnerClassExclusionStrategy  [METHOD] isStatic [RETURN_TYPE] boolean   Class<?> clazz [VARIABLES] boolean  Class  clazz  
