[BugLab_Variable_Misuse]^if ( longName!=null && !" ".equals ( shortName )  ) {^54^^^^^39^69^if ( shortName!=null && !" ".equals ( shortName )  ) {^[CLASS] CLI2Converter  [METHOD] option [RETURN_TYPE] Option   Option option1 [VARIABLES] boolean  ArgumentBuilder  abuilder  Option  option1  DefaultOptionBuilder  obuilder  Object  type  String  argName  description  longName  shortName  
[BugLab_Wrong_Operator]^if ( shortName!=null || !" ".equals ( shortName )  ) {^54^^^^^39^69^if ( shortName!=null && !" ".equals ( shortName )  ) {^[CLASS] CLI2Converter  [METHOD] option [RETURN_TYPE] Option   Option option1 [VARIABLES] boolean  ArgumentBuilder  abuilder  Option  option1  DefaultOptionBuilder  obuilder  Object  type  String  argName  description  longName  shortName  
[BugLab_Variable_Misuse]^obuilder.withShortName ( longName ) ;^55^^^^^40^70^obuilder.withShortName ( shortName ) ;^[CLASS] CLI2Converter  [METHOD] option [RETURN_TYPE] Option   Option option1 [VARIABLES] boolean  ArgumentBuilder  abuilder  Option  option1  DefaultOptionBuilder  obuilder  Object  type  String  argName  description  longName  shortName  
[BugLab_Variable_Misuse]^obuilder.withLongName ( shortName ) ;^60^^^^^45^75^obuilder.withLongName ( longName ) ;^[CLASS] CLI2Converter  [METHOD] option [RETURN_TYPE] Option   Option option1 [VARIABLES] boolean  ArgumentBuilder  abuilder  Option  option1  DefaultOptionBuilder  obuilder  Object  type  String  argName  description  longName  shortName  
[BugLab_Variable_Misuse]^if ( shortName!=null ) {^65^^^^^50^80^if ( description!=null ) {^[CLASS] CLI2Converter  [METHOD] option [RETURN_TYPE] Option   Option option1 [VARIABLES] boolean  ArgumentBuilder  abuilder  Option  option1  DefaultOptionBuilder  obuilder  Object  type  String  argName  description  longName  shortName  
[BugLab_Variable_Misuse]^obuilder.withDescription ( shortName ) ;^66^^^^^51^81^obuilder.withDescription ( description ) ;^[CLASS] CLI2Converter  [METHOD] option [RETURN_TYPE] Option   Option option1 [VARIABLES] boolean  ArgumentBuilder  abuilder  Option  option1  DefaultOptionBuilder  obuilder  Object  type  String  argName  description  longName  shortName  
[BugLab_Wrong_Literal]^abuilder.withMinimum ( -1 ) ;^78^^^^^63^93^abuilder.withMinimum ( 0 ) ;^[CLASS] CLI2Converter  [METHOD] option [RETURN_TYPE] Option   Option option1 [VARIABLES] boolean  ArgumentBuilder  abuilder  Option  option1  DefaultOptionBuilder  obuilder  Object  type  String  argName  description  longName  shortName  
[BugLab_Variable_Misuse]^abuilder.withName ( shortName ) ;^72^^^^^57^87^abuilder.withName ( argName ) ;^[CLASS] CLI2Converter  [METHOD] option [RETURN_TYPE] Option   Option option1 [VARIABLES] boolean  ArgumentBuilder  abuilder  Option  option1  DefaultOptionBuilder  obuilder  Object  type  String  argName  description  longName  shortName  
[BugLab_Wrong_Literal]^abuilder.withMinimum ( 1 ) ;^78^^^^^63^93^abuilder.withMinimum ( 0 ) ;^[CLASS] CLI2Converter  [METHOD] option [RETURN_TYPE] Option   Option option1 [VARIABLES] boolean  ArgumentBuilder  abuilder  Option  option1  DefaultOptionBuilder  obuilder  Object  type  String  argName  description  longName  shortName  
[BugLab_Variable_Misuse]^final Option option2 = option ( option2 ) ;^108^^^^^102^119^final Option option2 = option ( option1 ) ;^[CLASS] CLI2Converter  [METHOD] group [RETURN_TYPE] Group   OptionGroup optionGroup1 [VARIABLES] OptionGroup  optionGroup1  boolean  GroupBuilder  gbuilder  Iterator  i  Option  option1  option2  
[BugLab_Wrong_Literal]^gbuilder.withMaximum ( 0 ) ;^112^^^^^102^119^gbuilder.withMaximum ( 1 ) ;^[CLASS] CLI2Converter  [METHOD] group [RETURN_TYPE] Group   OptionGroup optionGroup1 [VARIABLES] OptionGroup  optionGroup1  boolean  GroupBuilder  gbuilder  Iterator  i  Option  option1  option2  
[BugLab_Wrong_Literal]^if ( optionGroup2.isRequired (  )  ) {^114^^^^^102^119^if ( optionGroup1.isRequired (  )  ) {^[CLASS] CLI2Converter  [METHOD] group [RETURN_TYPE] Group   OptionGroup optionGroup1 [VARIABLES] OptionGroup  optionGroup1  boolean  GroupBuilder  gbuilder  Iterator  i  Option  option1  option2  
[BugLab_Variable_Misuse]^final Option option2 = option ( option2 ) ;^143^^^^^127^149^final Option option2 = option ( option1 ) ;^[CLASS] CLI2Converter  [METHOD] group [RETURN_TYPE] Group   Options options1 [VARIABLES] Options  options1  Group  group  Set  optionGroups  OptionGroup  optionGroup1  boolean  GroupBuilder  gbuilder  Iterator  i  Option  option1  option2  
[BugLab_Variable_Misuse]^gbuilder.withOption ( option1 ) ;^144^^^^^127^149^gbuilder.withOption ( option2 ) ;^[CLASS] CLI2Converter  [METHOD] group [RETURN_TYPE] Group   Options options1 [VARIABLES] Options  options1  Group  group  Set  optionGroups  OptionGroup  optionGroup1  boolean  GroupBuilder  gbuilder  Iterator  i  Option  option1  option2  
[BugLab_Variable_Misuse]^if ( !optionInAGroup ( option2,optionGroups )  ) {^142^^^^^127^149^if ( !optionInAGroup ( option1,optionGroups )  ) {^[CLASS] CLI2Converter  [METHOD] group [RETURN_TYPE] Group   Options options1 [VARIABLES] Options  options1  Group  group  Set  optionGroups  OptionGroup  optionGroup1  boolean  GroupBuilder  gbuilder  Iterator  i  Option  option1  option2  
[BugLab_Argument_Swapping]^if ( option1.getOptions (  ) .contains ( group )  ) {^154^^^^^151^159^if ( group.getOptions (  ) .contains ( option1 )  ) {^[CLASS] CLI2Converter  [METHOD] optionInAGroup [RETURN_TYPE] boolean   Option option1 Set optionGroups [VARIABLES] Set  optionGroups  OptionGroup  group  boolean  Iterator  i  Option  option1  
[BugLab_Wrong_Literal]^return false;^155^^^^^151^159^return true;^[CLASS] CLI2Converter  [METHOD] optionInAGroup [RETURN_TYPE] boolean   Option option1 Set optionGroups [VARIABLES] Set  optionGroups  OptionGroup  group  boolean  Iterator  i  Option  option1  
[BugLab_Argument_Swapping]^for  ( Iterator optionGroups = i.iterator (  ) ; i.hasNext (  ) ; )  {^152^^^^^151^159^for  ( Iterator i = optionGroups.iterator (  ) ; i.hasNext (  ) ; )  {^[CLASS] CLI2Converter  [METHOD] optionInAGroup [RETURN_TYPE] boolean   Option option1 Set optionGroups [VARIABLES] Set  optionGroups  OptionGroup  group  boolean  Iterator  i  Option  option1  
[BugLab_Wrong_Literal]^return true;^158^^^^^151^159^return false;^[CLASS] CLI2Converter  [METHOD] optionInAGroup [RETURN_TYPE] boolean   Option option1 Set optionGroups [VARIABLES] Set  optionGroups  OptionGroup  group  boolean  Iterator  i  Option  option1  
[BugLab_Variable_Misuse]^if ( type==null ) {^184^^^^^179^189^if ( converted==null ) {^[CLASS] TypeHandlerValidator  [METHOD] validate [RETURN_TYPE] void   List values [VARIABLES] Object  converted  type  List  values  String  value  boolean  ListIterator  i  
[BugLab_Variable_Misuse]^final Object converted = TypeHandler.createValue ( value,converted ) ;^183^^^^^179^189^final Object converted = TypeHandler.createValue ( value,type ) ;^[CLASS] TypeHandlerValidator  [METHOD] validate [RETURN_TYPE] void   List values [VARIABLES] Object  converted  type  List  values  String  value  boolean  ListIterator  i  
[BugLab_Variable_Misuse]^i.set ( type ) ;^187^^^^^179^189^i.set ( converted ) ;^[CLASS] TypeHandlerValidator  [METHOD] validate [RETURN_TYPE] void   List values [VARIABLES] Object  converted  type  List  values  String  value  boolean  ListIterator  i  
