[BugLab_Wrong_Operator]^if  ( s1 != null )  {^73^^^^^58^88^if  ( s1 == null )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^return  ( s1 == null ) ;^74^^^^^59^89^return  ( s2 == null ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^return  ( s2 != null ) ;^74^^^^^59^89^return  ( s2 == null ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^if  ( s1 == null )  {^76^^^^^61^91^if  ( s2 == null )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^if  ( s2 != null )  {^76^^^^^61^91^if  ( s2 == null )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Literal]^return true;^77^^^^^62^92^return false;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^AttributedCharacterIterator it1 = s2.getIterator (  ) ;^79^^^^^64^94^AttributedCharacterIterator it1 = s1.getIterator (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^AttributedCharacterIterator it2 = s1.getIterator (  ) ;^80^^^^^65^95^AttributedCharacterIterator it2 = s2.getIterator (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^char c1 = it2.first (  ) ;^81^^^^^66^96^char c1 = it1.first (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^char c2 = it1.first (  ) ;^82^^^^^67^97^char c2 = it2.first (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Literal]^int start = limit2;^83^^^^^68^98^int start = 0;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^while  ( c2 != CharacterIterator.DONE )  {^84^^^^^69^99^while  ( c1 != CharacterIterator.DONE )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^while  ( c1 != CharacterIterator.c2 )  {^84^^^^^69^99^while  ( c1 != CharacterIterator.DONE )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^while  ( c1 == CharacterIterator.DONE )  {^84^^^^^69^99^while  ( c1 != CharacterIterator.DONE )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^while  ( c1 >= CharacterIterator.DONE )  {^84^^^^^69^99^while  ( c1 != CharacterIterator.DONE )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^if  ( start != limit2 )  {^87^^^^^72^102^if  ( limit1 != limit2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^if  ( limit1 != start )  {^87^^^^^72^102^if  ( limit1 != limit2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Argument_Swapping]^if  ( limit2 != limit1 )  {^87^^^^^72^102^if  ( limit1 != limit2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^if  ( limit1 == limit2 )  {^87^^^^^72^102^if  ( limit1 != limit2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Literal]^return true;^88^^^^^73^103^return false;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Literal]^return true;^94^^^^^79^109^return false;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Argument_Swapping]^if  ( c2 != c1 )  {^98^^^^^83^113^if  ( c1 != c2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^if  ( c1 == c2 )  {^98^^^^^83^113^if  ( c1 != c2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Literal]^return true;^99^^^^^84^114^return false;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^for  ( int i = limit2; i < limit1; i++ )  {^97^^^^^82^112^for  ( int i = start; i < limit1; i++ )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^for  ( startnt i = start; i < limit1; i++ )  {^97^^^^^82^112^for  ( int i = start; i < limit1; i++ )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^for  ( int i = start; i < start; i++ )  {^97^^^^^82^112^for  ( int i = start; i < limit1; i++ )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Argument_Swapping]^for  ( startnt i = i; i < limit1; i++ )  {^97^^^^^82^112^for  ( int i = start; i < limit1; i++ )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Argument_Swapping]^for  ( int i = limit1; i < start; i++ )  {^97^^^^^82^112^for  ( int i = start; i < limit1; i++ )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^for  ( int i = start; i == limit1; i++ )  {^97^^^^^82^112^for  ( int i = start; i < limit1; i++ )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^if  ( c1 >= c2 )  {^98^^^^^83^113^if  ( c1 != c2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^c1 = it2.next (  ) ;^101^^^^^86^116^c1 = it1.next (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^c2 = it1.next (  ) ;^102^^^^^87^117^c2 = it2.next (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^start = start;^104^^^^^89^119^start = limit1;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^int limit1 = it2.getRunLimit (  ) ;^85^^^^^70^100^int limit1 = it1.getRunLimit (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^int limit2 = it1.getRunLimit (  ) ;^86^^^^^71^101^int limit2 = it2.getRunLimit (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^Map m1 = it2.getAttributes (  ) ;^91^^^^^76^106^Map m1 = it1.getAttributes (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^Map m2 = it1.getAttributes (  ) ;^92^^^^^77^107^Map m2 = it2.getAttributes (  ) ;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^if  ( limit1 >= limit2 )  {^87^^^^^72^102^if  ( limit1 != limit2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^if  ( c1 > c2 )  {^98^^^^^83^113^if  ( c1 != c2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Argument_Swapping]^for  ( limit1nt i = start; i < i; i++ )  {^97^^^^^82^112^for  ( int i = start; i < limit1; i++ )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^for  ( int i = start; i <= limit1; i++ )  {^97^^^^^82^112^for  ( int i = start; i < limit1; i++ )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^if  ( c1 <= c2 )  {^98^^^^^83^113^if  ( c1 != c2 )  {^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Variable_Misuse]^return c1 == CharacterIterator.DONE;^106^^^^^91^121^return c2 == CharacterIterator.DONE;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  
[BugLab_Wrong_Operator]^return c2 <= CharacterIterator.DONE;^106^^^^^91^121^return c2 == CharacterIterator.DONE;^[CLASS] AttributedStringUtilities  [METHOD] equal [RETURN_TYPE] boolean   AttributedString s1 AttributedString s2 [VARIABLES] AttributedString  s1  s2  boolean  char  c1  c2  Map  m1  m2  int  i  limit1  limit2  start  AttributedCharacterIterator  it1  it2  