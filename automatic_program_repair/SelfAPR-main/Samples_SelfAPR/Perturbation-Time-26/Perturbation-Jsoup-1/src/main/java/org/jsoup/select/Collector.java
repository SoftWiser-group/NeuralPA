[P5_Replace_Variable]^accumulateMatches (  elements, root ) ;^23^^^^^21^25^accumulateMatches ( eval, elements, root ) ;^[CLASS] Collector  [METHOD] collect [RETURN_TYPE] Elements   Evaluator eval Element root [VARIABLES] boolean  Element  root  Evaluator  eval  Elements  elements  
[P5_Replace_Variable]^accumulateMatches ( eval,  root ) ;^23^^^^^21^25^accumulateMatches ( eval, elements, root ) ;^[CLASS] Collector  [METHOD] collect [RETURN_TYPE] Elements   Evaluator eval Element root [VARIABLES] boolean  Element  root  Evaluator  eval  Elements  elements  
[P5_Replace_Variable]^accumulateMatches ( eval, elements ) ;^23^^^^^21^25^accumulateMatches ( eval, elements, root ) ;^[CLASS] Collector  [METHOD] collect [RETURN_TYPE] Elements   Evaluator eval Element root [VARIABLES] boolean  Element  root  Evaluator  eval  Elements  elements  
[P5_Replace_Variable]^accumulateMatches ( elements, eval, root ) ;^23^^^^^21^25^accumulateMatches ( eval, elements, root ) ;^[CLASS] Collector  [METHOD] collect [RETURN_TYPE] Elements   Evaluator eval Element root [VARIABLES] boolean  Element  root  Evaluator  eval  Elements  elements  
[P5_Replace_Variable]^accumulateMatches ( root, elements, eval ) ;^23^^^^^21^25^accumulateMatches ( eval, elements, root ) ;^[CLASS] Collector  [METHOD] collect [RETURN_TYPE] Elements   Evaluator eval Element root [VARIABLES] boolean  Element  root  Evaluator  eval  Elements  elements  
[P14_Delete_Statement]^^23^24^^^^21^25^accumulateMatches ( eval, elements, root ) ; return elements;^[CLASS] Collector  [METHOD] collect [RETURN_TYPE] Elements   Evaluator eval Element root [VARIABLES] boolean  Element  root  Evaluator  eval  Elements  elements  
[P11_Insert_Donor_Statement]^accumulateMatches ( eval, elements, child ) ;accumulateMatches ( eval, elements, root ) ;^23^^^^^21^25^accumulateMatches ( eval, elements, root ) ;^[CLASS] Collector  [METHOD] collect [RETURN_TYPE] Elements   Evaluator eval Element root [VARIABLES] boolean  Element  root  Evaluator  eval  Elements  elements  
[P5_Replace_Variable]^if  ( eval.matches ( child )  ) elements.add ( element ) ;^28^29^^^^27^32^if  ( eval.matches ( element )  ) elements.add ( element ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^if  ( elements.matches ( element )  ) eval.add ( element ) ;^28^29^^^^27^32^if  ( eval.matches ( element )  ) elements.add ( element ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^if  ( element.matches ( eval )  ) elements.add ( element ) ;^28^29^^^^27^32^if  ( eval.matches ( element )  ) elements.add ( element ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P15_Unwrap_Block]^elements.add(element);^28^29^30^31^32^27^32^if  ( eval.matches ( element )  ) elements.add ( element ) ; for  ( Element child: element.children (  )  ) accumulateMatches ( eval, elements, child ) ; }^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P16_Remove_Block]^^28^29^30^31^32^27^32^if  ( eval.matches ( element )  ) elements.add ( element ) ; for  ( Element child: element.children (  )  ) accumulateMatches ( eval, elements, child ) ; }^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^elements.add ( child ) ;^29^^^^^27^32^elements.add ( element ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P14_Delete_Statement]^^29^^^^^27^32^elements.add ( element ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^for  ( Element child: child.children (  )  ) accumulateMatches ( eval, elements, child ) ;^30^31^^^^27^32^for  ( Element child: element.children (  )  ) accumulateMatches ( eval, elements, child ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P14_Delete_Statement]^^30^31^32^^^27^32^for  ( Element child: element.children (  )  ) accumulateMatches ( eval, elements, child ) ; }^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^accumulateMatches ( eval, elements, element ) ;^31^^^^^27^32^accumulateMatches ( eval, elements, child ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^accumulateMatches (  elements, child ) ;^31^^^^^27^32^accumulateMatches ( eval, elements, child ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^accumulateMatches ( eval,  child ) ;^31^^^^^27^32^accumulateMatches ( eval, elements, child ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^accumulateMatches ( eval, elements ) ;^31^^^^^27^32^accumulateMatches ( eval, elements, child ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^accumulateMatches ( elements, eval, child ) ;^31^^^^^27^32^accumulateMatches ( eval, elements, child ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P5_Replace_Variable]^accumulateMatches ( eval, child, elements ) ;^31^^^^^27^32^accumulateMatches ( eval, elements, child ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P14_Delete_Statement]^^31^^^^^27^32^accumulateMatches ( eval, elements, child ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
[P11_Insert_Donor_Statement]^accumulateMatches ( eval, elements, root ) ;accumulateMatches ( eval, elements, child ) ;^31^^^^^27^32^accumulateMatches ( eval, elements, child ) ;^[CLASS] Collector  [METHOD] accumulateMatches [RETURN_TYPE] void   Evaluator eval Element> elements Element element [VARIABLES] boolean  Element  child  element  List  elements  Evaluator  eval  
