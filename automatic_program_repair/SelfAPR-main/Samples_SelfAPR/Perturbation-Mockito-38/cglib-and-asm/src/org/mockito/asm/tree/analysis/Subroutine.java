[P1_Replace_Type]^Set  callers;^49^^^^^44^54^List callers;^[CLASS] Subroutine   [VARIABLES] 
[P8_Replace_Mix]^ArrayList  callers;^49^^^^^44^54^List callers;^[CLASS] Subroutine   [VARIABLES] 
[P8_Replace_Mix]^this.start =  null;^59^^^^^54^63^this.start = start;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P11_Insert_Donor_Statement]^result.start = start;this.start = start;^59^^^^^54^63^this.start = start;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P8_Replace_Mix]^this.access =  new boolean[null];^60^^^^^54^63^this.access = new boolean[maxLocals];^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P11_Insert_Donor_Statement]^result.access = new boolean[access.length];this.access = new boolean[maxLocals];^60^^^^^54^63^this.access = new boolean[maxLocals];^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P1_Replace_Type]^this.callers = new Array LinkedHashSet  (  ) ;^61^^^^^54^63^this.callers = new ArrayList (  ) ;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P4_Replace_Constructor]^this.callers = this.callers =  new ArrayList ( callers )  ;^61^^^^^54^63^this.callers = new ArrayList (  ) ;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P8_Replace_Mix]^this.callers =  new ArrayList ( callers )  ;^61^^^^^54^63^this.callers = new ArrayList (  ) ;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P11_Insert_Donor_Statement]^result.callers = new ArrayList ( callers ) ;this.callers = new ArrayList (  ) ;^61^^^^^54^63^this.callers = new ArrayList (  ) ;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P1_Replace_Type]^this.callers = new  LinkedList  (  ) ;^61^^^^^54^63^this.callers = new ArrayList (  ) ;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P7_Replace_Invocation]^callers.get ( caller ) ;^62^^^^^54^63^callers.add ( caller ) ;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P14_Delete_Statement]^^62^^^^^54^63^callers.add ( caller ) ;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P12_Insert_Condition]^if  ( !callers.contains ( caller )  )  { callers.add ( caller ) ; }^62^^^^^54^63^callers.add ( caller ) ;^[CLASS] Subroutine  [METHOD] <init> [RETURN_TYPE] JumpInsnNode)   LabelNode start final int maxLocals JumpInsnNode caller [VARIABLES] LabelNode  start  JumpInsnNode  caller  List  callers  boolean[]  access  boolean  int  maxLocals  
[P8_Replace_Mix]^result.start =  start;^67^^^^^65^72^result.start = start;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P11_Insert_Donor_Statement]^this.start = start;result.start = start;^67^^^^^65^72^result.start = start;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P8_Replace_Mix]^result.access =  new boolean[access.length];^68^^^^^65^72^result.access = new boolean[access.length];^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P11_Insert_Donor_Statement]^this.access = new boolean[maxLocals];result.access = new boolean[access.length];^68^^^^^65^72^result.access = new boolean[access.length];^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P3_Replace_Literal]^System.arraycopy ( access, 2, result.access, 2, access.length ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P3_Replace_Literal]^System.arraycopy ( access, 1, result.access, 1, access.length ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P5_Replace_Variable]^System.arraycopy ( access, 0, access, 0, access.length ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P5_Replace_Variable]^System.arraycopy ( access, 0.access, 0, access.length ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P5_Replace_Variable]^System.arraycopy (  0, result.access, 0, access.length ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P5_Replace_Variable]^System.arraycopy ( access, 0,  0, access.length ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P5_Replace_Variable]^System.arraycopy ( access, 0, result.access, 0 ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P5_Replace_Variable]^System.arraycopy ( result, 0, access.access, 0, access.length ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P5_Replace_Variable]^System.arraycopy ( result.access, 0, access, 0, access.length ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P5_Replace_Variable]^System.arraycopy ( access.length, 0, result.access, 0, access ) ;^69^^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P14_Delete_Statement]^^69^70^^^^65^72^System.arraycopy ( access, 0, result.access, 0, access.length ) ; result.callers = new ArrayList ( callers ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P1_Replace_Type]^result.callers = new Array ArrayList  ( callers ) ;^70^^^^^65^72^result.callers = new ArrayList ( callers ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P4_Replace_Constructor]^result.callers = result.callers =  new ArrayList (  )  ;^70^^^^^65^72^result.callers = new ArrayList ( callers ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P8_Replace_Mix]^result.callers =  new ArrayList ( callers ) ;^70^^^^^65^72^result.callers = new ArrayList ( callers ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P11_Insert_Donor_Statement]^this.callers = new ArrayList (  ) ;result.callers = new ArrayList ( callers ) ;^70^^^^^65^72^result.callers = new ArrayList ( callers ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P1_Replace_Type]^result.callers = new  List  ( callers ) ;^70^^^^^65^72^result.callers = new ArrayList ( callers ) ;^[CLASS] Subroutine  [METHOD] copy [RETURN_TYPE] Subroutine   [VARIABLES] LabelNode  start  List  callers  boolean[]  access  boolean  Subroutine  result  
[P3_Replace_Literal]^boolean changes = true;^75^^^^^74^92^boolean changes = false;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P2_Replace_Operator]^if  ( subroutine.access[i] || !access[i] )  {^77^^^^^74^92^if  ( subroutine.access[i] && !access[i] )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^if  ( subroutine[i] && !access[i] )  {^77^^^^^74^92^if  ( subroutine.access[i] && !access[i] )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P6_Replace_Expression]^if  ( subroutine.access[i] ) {^77^^^^^74^92^if  ( subroutine.access[i] && !access[i] )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P6_Replace_Expression]^if  (  !access[i] )  {^77^^^^^74^92^if  ( subroutine.access[i] && !access[i] )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P15_Unwrap_Block]^access[i] = true; changes = true;^77^78^79^80^^74^92^if  ( subroutine.access[i] && !access[i] )  { access[i] = true; changes = true; }^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P16_Remove_Block]^^77^78^79^80^^74^92^if  ( subroutine.access[i] && !access[i] )  { access[i] = true; changes = true; }^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^access[i] = false;^78^^^^^74^92^access[i] = true;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P11_Insert_Donor_Statement]^changes = true;access[i] = true;^78^^^^^74^92^access[i] = true;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^changes = false;^79^^^^^74^92^changes = true;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P11_Insert_Donor_Statement]^access[i] = true;changes = true;^79^^^^^74^92^changes = true;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P1_Replace_Type]^for  (  short  i = 0; i < access.length; ++i )  {^76^^^^^74^92^for  ( int i = 0; i < access.length; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P2_Replace_Operator]^for  ( int i = 0; i <= access.length; ++i )  {^76^^^^^74^92^for  ( int i = 0; i < access.length; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = i; i < access.length; ++i )  {^76^^^^^74^92^for  ( int i = 0; i < access.length; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^for  ( access.lengthnt i = 0; i < i; ++i )  {^76^^^^^74^92^for  ( int i = 0; i < access.length; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^for  ( accessnt i = 0; i < i.length; ++i )  {^76^^^^^74^92^for  ( int i = 0; i < access.length; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^if  ( subroutine.access.access[i] && !access[i] )  {^77^^^^^74^92^if  ( subroutine.access[i] && !access[i] )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P8_Replace_Mix]^if  ( access[i] ) {^77^^^^^74^92^if  ( subroutine.access[i] && !access[i] )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P1_Replace_Type]^for  (  long  i = 0; i < access.length; ++i )  {^76^^^^^74^92^for  ( int i = 0; i < access.length; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P2_Replace_Operator]^if  ( subroutine.start >= start )  {^82^^^^^74^92^if  ( subroutine.start == start )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^if  ( start == start )  {^82^^^^^74^92^if  ( subroutine.start == start )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^if  ( subroutine == start )  {^82^^^^^74^92^if  ( subroutine.start == start )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P6_Replace_Expression]^if  ( i < size() )  {^82^^^^^74^92^if  ( subroutine.start == start )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P7_Replace_Invocation]^if  ( !callers.get ( caller )  )  {^85^^^^^74^92^if  ( !callers.contains ( caller )  )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P15_Unwrap_Block]^callers.add(caller); changes = true;^85^86^87^88^^74^92^if  ( !callers.contains ( caller )  )  { callers.add ( caller ) ; changes = true; }^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P16_Remove_Block]^^85^86^87^88^^74^92^if  ( !callers.contains ( caller )  )  { callers.add ( caller ) ; changes = true; }^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^changes = false;^87^^^^^74^92^changes = true;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P11_Insert_Donor_Statement]^access[i] = true;changes = true;^87^^^^^74^92^changes = true;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P7_Replace_Invocation]^callers.get ( caller ) ;^86^^^^^74^92^callers.add ( caller ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P7_Replace_Invocation]^callers .contains ( caller )  ;^86^^^^^74^92^callers.add ( caller ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P14_Delete_Statement]^^86^^^^^74^92^callers.add ( caller ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P1_Replace_Type]^for  (  short  i = 0; i < subroutine.callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P2_Replace_Operator]^for  ( int i = 0; i <= subroutine.callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = 5; i < subroutine.callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = 0; i < subroutine.callers.size() + 4 ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^for  ( int i = 0; i < callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^for  ( subroutinent i = 0; i < i.callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P7_Replace_Invocation]^for  ( int i = 0; i < subroutine.callers.add (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P13_Insert_Block]^if  (  ( subroutine.start )  ==  ( start )  )  {     for  ( int i = 0; i <  ( size (  )  ) ; ++i )  {         Object caller = get ( i ) ;         if  ( ! ( callers.contains ( caller )  )  )  {             callers.add ( caller ) ;             changes = true;         }     } }^83^^^^^74^92^[Delete]^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P14_Delete_Statement]^^86^87^^^^74^92^callers.add ( caller ) ; changes = true;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P12_Insert_Condition]^if  ( !callers.contains ( caller )  )  { callers.add ( caller ) ; }^86^^^^^74^92^callers.add ( caller ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^Object caller = i.callers.get ( subroutine ) ;^84^^^^^74^92^Object caller = subroutine.callers.get ( i ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^Object caller = subroutine.get ( i ) ;^84^^^^^74^92^Object caller = subroutine.callers.get ( i ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P8_Replace_Mix]^Object caller = callers.get ( i ) ;^84^^^^^74^92^Object caller = subroutine.callers.get ( i ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^Object caller = subroutine.callers.callers.get ( i ) ;^84^^^^^74^92^Object caller = subroutine.callers.get ( i ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^Object caller = i.get ( subroutine.callers ) ;^84^^^^^74^92^Object caller = subroutine.callers.get ( i ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P14_Delete_Statement]^^84^^^^^74^92^Object caller = subroutine.callers.get ( i ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = i; i < subroutine.callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = 0; i < subroutine.callers.size() + 5 ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P7_Replace_Invocation]^Object caller = subroutine.callers .contains ( caller )  ;^84^^^^^74^92^Object caller = subroutine.callers.get ( i ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = 0; i < subroutine.callers.size() + 2 ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^for  ( int i = 0; i < subroutine.callers.callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P5_Replace_Variable]^for  ( int i = 0; i < subroutine.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P14_Delete_Statement]^^83^84^85^86^87^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  { Object caller = subroutine.callers.get ( i ) ; if  ( !callers.contains ( caller )  )  { callers.add ( caller ) ; changes = true; }^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P1_Replace_Type]^for  (  long  i = 0; i < subroutine.callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P2_Replace_Operator]^for  ( int i = 0; i > subroutine.callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = 0; i < subroutine.callers.size() + 8 ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P7_Replace_Invocation]^for  ( int i = 0; i < subroutine.callers .contains ( caller )  ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = 4; i < subroutine.callers.size (  ) ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = 0; i < subroutine.callers.size() + 1 ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P8_Replace_Mix]^Object caller = callers .contains ( caller )  ;^84^^^^^74^92^Object caller = subroutine.callers.get ( i ) ;^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
[P3_Replace_Literal]^for  ( int i = 0; i < subroutine.callers.size() - 7 ; ++i )  {^83^^^^^74^92^for  ( int i = 0; i < subroutine.callers.size (  ) ; ++i )  {^[CLASS] Subroutine  [METHOD] merge [RETURN_TYPE] boolean   Subroutine subroutine [VARIABLES] LabelNode  start  boolean  changes  Subroutine  subroutine  List  callers  Object  caller  boolean[]  access  int  i  
